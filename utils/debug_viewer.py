# debug_viewer.py
"""
Utility script to view and manage location editor debug files
Run this script via Railway SSH to see available debug files
"""

import os
import json
from datetime import datetime

DEBUG_DIR = "/tmp/location_debug"

def list_debug_files():
    """List all debug files with timestamps"""
    if not os.path.exists(DEBUG_DIR):
        print(f"Debug directory {DEBUG_DIR} does not exist")
        return

    files = []
    for filename in os.listdir(DEBUG_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(DEBUG_DIR, filename)
            mtime = os.path.getmtime(filepath)
            size = os.path.getsize(filepath)
            files.append((filename, filepath, mtime, size))

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[2], reverse=True)

    print(f"\n📁 Debug files in {DEBUG_DIR}:")
    print("=" * 80)

    for filename, filepath, mtime, size in files:
        timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = size / 1024
        print(f"{filename}")
        print(f"  📅 {timestamp} | 📊 {size_kb:.1f} KB | 📂 {filepath}")
        print()

def view_file_summary(filename):
    """View summary of a debug file"""
    filepath = os.path.join(DEBUG_DIR, filename)

    if not os.path.exists(filepath):
        print(f"File {filepath} not found")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n📋 Summary of {filename}:")
        print("=" * 60)

        # Basic info
        if 'timestamp' in data:
            print(f"🕐 Timestamp: {data['timestamp']}")
        if 'user_query' in data:
            print(f"🔍 User Query: {data['user_query']}")
        if 'location_description' in data:
            print(f"📍 Location: {data['location_description']}")

        # Summary section
        if 'summary' in data:
            print(f"\n📊 Data Summary:")
            for key, value in data['summary'].items():
                print(f"  {key}: {value}")

        # Selection summary
        if 'selection_summary' in data:
            print(f"\n🎯 Selection Summary:")
            for key, value in data['selection_summary'].items():
                print(f"  {key}: {value}")

        # Generation summary  
        if 'generation_summary' in data:
            print(f"\n📝 Generation Summary:")
            for key, value in data['generation_summary'].items():
                print(f"  {key}: {value}")

        # Show first restaurant as example
        if 'combined_venues' in data and data['combined_venues']:
            venue = data['combined_venues'][0]
            print(f"\n🍽️ Example Venue:")
            print(f"  Name: {venue.get('name', 'N/A')}")
            print(f"  Rating: {venue.get('rating', 'N/A')}")
            print(f"  Reviews count: {len(venue.get('google_reviews', []))}")
            print(f"  Media coverage: {venue.get('has_professional_coverage', False)}")

        # Show generated descriptions
        if 'generated_descriptions' in data and data['generated_descriptions']:
            print(f"\n📖 Generated Descriptions:")
            for i, desc in enumerate(data['generated_descriptions'][:3], 1):
                name = desc.get('name', 'Unknown')
                description = desc.get('description', 'No description')
                print(f"  {i}. {name}")
                print(f"     {description[:100]}...")
                print()

    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    """Main function for interactive use"""
    print("🔍 Location Editor Debug File Viewer")
    print("=" * 50)

    if len(os.sys.argv) > 1:
        # View specific file
        filename = os.sys.argv[1]
        view_file_summary(filename)
    else:
        # List all files
        list_debug_files()

        print("\n💡 Usage:")
        print(f"  python debug_viewer.py                    # List all files")
        print(f"  python debug_viewer.py filename.json     # View file summary")
        print(f"  cat {DEBUG_DIR}/filename.json | head -50  # View raw file")

if __name__ == "__main__":
    main()