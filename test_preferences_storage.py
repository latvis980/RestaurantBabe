#!/usr/bin/env python3
"""
Test script to diagnose user preferences storage issue
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_preferences_storage():
    """Test the full preferences storage flow"""

    print("\n" + "="*60)
    print("TESTING USER PREFERENCES STORAGE")
    print("="*60 + "\n")

    try:
        # Step 1: Import config
        print("1️⃣ Importing config...")
        import config
        print(f"   ✅ Memory store type: {config.MEMORY_STORE_TYPE}")
        print(f"   ✅ Auto update preferences: {config.AUTO_UPDATE_USER_PREFERENCES}")

        # Step 2: Import memory system
        print("\n2️⃣ Importing AIMemorySystem...")
        from utils.ai_memory_system import AIMemorySystem, UserPreferences
        memory_system = AIMemorySystem(config)
        print(f"   ✅ Memory system initialized (persistent: {memory_system.is_persistent})")

        # Step 3: Test getting preferences
        test_user_id = 12345
        print(f"\n3️⃣ Testing get_user_preferences for user {test_user_id}...")
        prefs = await memory_system.get_user_preferences(test_user_id)
        print(f"   ✅ Current preferences:")
        print(f"      Cities: {prefs.preferred_cities}")
        print(f"      Cuisines: {prefs.preferred_cuisines}")
        print(f"      Budget: {prefs.budget_range}")

        # Step 4: Update preferences
        print(f"\n4️⃣ Updating preferences - adding 'italian' cuisine...")
        if 'italian' not in prefs.preferred_cuisines:
            prefs.preferred_cuisines.append('italian')
        if 'lisbon' not in prefs.preferred_cities:
            prefs.preferred_cities.append('lisbon')

        result = await memory_system.update_user_preferences(test_user_id, prefs)
        print(f"   {'✅' if result else '❌'} Update result: {result}")

        # Step 5: Verify update
        print(f"\n5️⃣ Verifying update...")
        updated_prefs = await memory_system.get_user_preferences(test_user_id)
        print(f"   Updated preferences:")
        print(f"      Cities: {updated_prefs.preferred_cities}")
        print(f"      Cuisines: {updated_prefs.preferred_cuisines}")

        if 'italian' in updated_prefs.preferred_cuisines:
            print(f"   ✅ Preferences successfully stored!")
        else:
            print(f"   ❌ Preferences NOT stored - check database connection")

        # Step 6: Test direct Supabase connection
        print(f"\n6️⃣ Testing direct Supabase query...")
        if memory_system.is_persistent:
            conn = memory_system.memory_store._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM user_preferences WHERE user_id = %s", (test_user_id,))
            result = cursor.fetchone()

            if result:
                print(f"   ✅ Found record in database:")
                print(f"      User ID: {result[1]}")
                print(f"      Cuisines: {result[3]}")
                print(f"      Cities: {result[2]}")
            else:
                print(f"   ❌ No record found in database for user {test_user_id}")

            cursor.close()
            memory_system.memory_store._return_connection(conn)

        print("\n" + "="*60)
        print("TEST COMPLETED")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_preferences_storage())
    sys.exit(0 if success else 1)