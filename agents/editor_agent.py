from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import tracing_v2_enabled
import json
from utils.debug_utils import dump_chain_state, log_function_call

class EditorAgent:
    def __init__(self, config):
        self.model = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0.3
        )

        # Editor prompt - updated to generate hidden_gems from main_list
        self.system_prompt = """
        You are a professional editor for a food publication specializing in restaurant recommendations.
        You will receive raw recommendations and must format & polish them according to the guidelines below.
        ⚠️ NEVER omit a restaurant: format every item you receive.

        ────────────────────────
        MANDATORY FIELDS (each restaurant)
        ────────────────────────
        • **Name** (bold)  
        • Street address ─ street number + street name  
        • 2-40-word informative description  
        • Price range (€, €€, €€€)  
        • Recommended dishes ─ list at least 2–3 signature items  
        • Sources ─ list every source you have; show ≥ 2 if available  
          – Do NOT cite Tripadvisor, Yelp, or Google  
        • missing_info ─ array of any mandatory fields still missing

        ────────────────────────
        OPTIONAL FIELDS (include when found)
        ────────────────────────
        • reservations_required (boolean) ─ clearly state if reservations are strongly advised  
        • instagram ─ "instagram.com/username"  
        • chef ─ name / background  
        • hours ─ opening hours  
        • atmosphere ─ noteworthy ambience details

        ────────────────────────
        MISSING-INFO POLICY
        ────────────────────────
        If mandatory data is missing, KEEP the restaurant in main_list  
        and list the absent fields in its missing_info array. Never move or delete an entry.

        ────────────────────────
        OUTPUT FORMAT
        ────────────────────────
        Return a single JSON object:
        {
          "formatted_recommendations": {
            "main_list":    [ …restaurants… ],
            "hidden_gems":  [ …restaurants… ]
          }
        }

        Each restaurant object:
        {
          "name": "<bold restaurant name>",
          "address": "<full street address | 'Address unavailable'>",
          "description": "<concise description>",
          "price_range": "€ / €€ / €€€",
          "recommended_dishes": ["dish1", "dish2", …],
          "sources": ["source1", "source2", …],
          "missing_info": ["fieldA", "fieldB", …],      # empty [] if none
          "reservations_required": true | false | null,
          "instagram": "instagram.com/username" | null,
          "chef": "Chef Name" | null,
          "hours": "Mon–Sun 12-22" | null,
          "atmosphere": "short ambience note" | null
        }

        ────────────────────────
        PRESENTATION RULES
        ────────────────────────
        1. Split output into two sections: "Recommended Restaurants" and "Hidden Gems".
        2. Apply consistent formatting; no emojis.
        3. Ensure every description is concise yet informative.
        4. Verify all mandatory data; flag what’s missing in missing_info.
        5. Also generate follow-up search queries (outside the JSON) for any missing mandatory info.
        """


        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Here are the restaurant recommendations to format:\n\n{recommendations}\n\nOriginal query: {original_query}")
        ])

        # Create chain
        self.chain = self.prompt | self.model

        self.config = config

    @log_function_call
    def edit(self, recommendations: dict, original_query: str) -> dict:
        """
        Format and polish the restaurant recommendations.

        Args
        ----
        recommendations : dict   – raw result from ListAnalyzer
        original_query  : str    – user’s initial query

        Returns
        -------
        dict with keys:
            formatted_recommendations : { main_list, hidden_gems }
            follow_up_queries         : list
        """
        with tracing_v2_enabled(project_name="restaurant-recommender"):
            try:
                # ── 1. debug input ──────────────────────────────────────────
                dump_chain_state(
                    "editor_input",
                    {
                        "recommendations_keys": list(recommendations.keys())
                        if isinstance(recommendations, dict)
                        else [],
                        "original_query": original_query,
                    },
                )

                # ── 2. follow-up query skeletons ──────────────────────────
                follow_up_queries = self._generate_follow_up_queries(
                    recommendations, original_query
                )

                # ── 3. guarantee structure presence ───────────────────────
                if not recommendations or not isinstance(recommendations, dict):
                    recommendations = {"main_list": []}

                if "main_list" not in recommendations:
                    if "recommended" in recommendations:                   # legacy
                        recommendations["main_list"] = recommendations.pop("recommended")
                    else:
                        recommendations["main_list"] = []

                # ── 4. call the LLM formatter ─────────────────────────────
                response = self.chain.invoke(
                    {
                        "recommendations": json.dumps(
                            recommendations, ensure_ascii=False, indent=2
                        ),
                        "original_query": original_query,
                    }
                )

                # ── 5. clean markdown fences & parse JSON ─────────────────
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                dump_chain_state("editor_raw_response", {"raw": content[:1000]})
                formatted_results = json.loads(content)

                # ── 6. normalise keys (recommended ➜ main_list) ──────────
                if "formatted_recommendations" in formatted_results:
                    formatted_rec = formatted_results["formatted_recommendations"]
                    if isinstance(formatted_rec, dict) and "recommended" in formatted_rec:
                        formatted_rec.setdefault("main_list", formatted_rec.pop("recommended"))
                else:
                    formatted_results = {
                        "formatted_recommendations": formatted_results
                    }
                    fr = formatted_results["formatted_recommendations"]
                    if isinstance(fr, dict) and "recommended" in fr:
                        fr.setdefault("main_list", fr.pop("recommended"))

                # ── 7. SAFETY NET – never lose the main list ─────────────
                formatted_rec = formatted_results.get("formatted_recommendations", {})
                if (
                    isinstance(formatted_rec, dict)
                    and not formatted_rec.get("main_list")            # empty after LLM
                    and recommendations.get("main_list")              # we had entries
                ):
                    formatted_rec["main_list"] = recommendations["main_list"]
                    formatted_results["formatted_recommendations"] = formatted_rec

                # ── 8. attach follow-up queries & finish ────────────────
                formatted_results["follow_up_queries"] = follow_up_queries

                dump_chain_state(
                    "editor_final_results",
                    {
                        "formatted_keys": list(formatted_results.keys()),
                        "follow_up_queries": len(follow_up_queries),
                    },
                )
                return formatted_results

            # ── 9. JSON or attr error inside try-block ───────────────────
            except (json.JSONDecodeError, AttributeError) as exc:
                dump_chain_state(
                    "editor_json_error",
                    {"error": str(exc), "response_preview": str(response.content)[:500]},
                )
                return {
                    "formatted_recommendations": {
                        "main_list": recommendations.get("main_list", []),
                        "hidden_gems": [],
                    },
                    "follow_up_queries": follow_up_queries if "follow_up_queries" in locals() else [],
                }

            # ── 10. any other exception ───────────────────────────────────
            except Exception as exc:
                dump_chain_state(
                    "editor_general_error",
                    {
                        "error": str(exc),
                        "recommendations_preview": str(recommendations)[:500],
                    },
                    error=exc,
                )
                return {
                    "formatted_recommendations": {
                        "main_list": recommendations.get("main_list", [])
                        if isinstance(recommendations, dict)
                        else [],
                        "hidden_gems": [],
                    },
                    "follow_up_queries": [],
                }


    # Improved follow-up query generator for EditorAgent

    def _generate_follow_up_queries(self, recommendations, original_query):
        """Generate follow-up search queries for each restaurant, focusing on mandatory information."""
        try:
            # 1. Turn recommendations into a JSON string once.
            rec_json = json.dumps(recommendations, ensure_ascii=False, indent=2)

            # 2. Prompt template *with* placeholders, no raw braces.
            follow_up_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 """
            You are an expert at crafting targeted web-search queries for restaurants.

            For each restaurant, create queries **only** to retrieve missing **MANDATORY** info:
            1. Address
            2. Price range
            3. Recommended dishes
            4. Reputable sources (Michelin, Time Out, etc.)

            Do **NOT** create queries for optional data (chef, Instagram, hours, atmosphere).

            If a restaurant already has all mandatory info, return just **one** query
            to check for mentions in respected guides.

            Return a JSON array, e.g.:
            [
              {{
                "restaurant_name": "Example Bistro",
                "queries": ["example bistro address", "example bistro price range"]
              }},
              …
            ]
            (Max 3 queries per restaurant.)
            """),
                ("human",
                 """
            Original user query:
            {original_query}

            Restaurant recommendations (JSON):
            {recommendations}

            Create the follow-up search queries as specified above.
            """)
            ])


            follow_up_chain = follow_up_prompt | self.model

            # 3. Supply the variables the template expects.
            response = follow_up_chain.invoke({
                "original_query": original_query,
                "recommendations": rec_json
            })

            # 4. Strip any markdown fences, then parse.
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)

        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"Error parsing follow-up queries: {exc}")
            return self._generate_basic_queries(recommendations)

        except Exception as exc:
            print(f"Error generating follow-up queries: {exc}")
            return self._generate_basic_queries(recommendations)


    def _generate_basic_queries(self, recommendations):
        """Generate basic follow-up queries focused on mandatory information if the main generation fails"""
        basic_queries = []

        # Get restaurants from main_list
        main_list = []
        if isinstance(recommendations, dict):
            if "main_list" in recommendations:
                main_list = recommendations["main_list"]
            elif "recommended" in recommendations:
                main_list = recommendations["recommended"]

        for restaurant in main_list:
            name = restaurant.get("name", "")
            if name:
                # Check what mandatory information is missing
                address = restaurant.get("address", "")
                price_range = restaurant.get("price_range", "")
                recommended_dishes = restaurant.get("recommended_dishes", [])
                sources = restaurant.get("sources", [])

                queries = []

                # Only create queries for missing mandatory information
                if not address or address == "Address unavailable":
                    queries.append(f"{name} restaurant address location")

                if not price_range:
                    queries.append(f"{name} restaurant price range cost")

                if not recommended_dishes or len(recommended_dishes) < 2:
                    queries.append(f"{name} restaurant signature dishes menu specialties")

                if not sources or len(sources) < 2:
                    queries.append(f"{name} restaurant reviews guide recommended by")

                # If no missing information or we have room for another query, add a guide check
                if not queries or len(queries) < 3:
                    queries.append(f"{name} restaurant michelin guide world's 50 best")

                # Limit to 3 queries maximum
                basic_queries.append({
                    "restaurant_name": name,
                    "queries": queries[:3]
                })

        return basic_queries