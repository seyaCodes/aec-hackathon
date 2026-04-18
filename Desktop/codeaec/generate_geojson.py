#!/usr/bin/env python3
"""
Generate GeoJSON for Algeria Wilayas with RPA zones and decision data.
Creates interactive GIS boundaries from decision_output_final.csv
"""

import json
import pandas as pd
from collections import defaultdict

# Wilaya centers (lat, lon) - used to create approximate boundaries
WILAYA_CENTERS = {
    'ALGER': (36.7538, 3.0588),
    'BLIDA': (36.4744, 2.8277),
    'BOUMERDES': (36.7611, 3.4801),
    'TIPAZA': (36.5915, 2.4508),
    'TIZI OUZOU': (36.7165, 4.0522),
    'CHLEF': (36.1716, 1.3389),
    'MEDEA': (36.2762, 2.7584),
    'AIN DEFLA': (36.2603, 1.9663),
    'SETIF': (36.1900, 5.4141),
    'BEJAIA': (36.7544, 5.0840),
    'CONSTANTINE': (36.3650, 6.6147),
    'ANNABA': (36.9000, 7.7600),
    'ORAN': (35.6984, -0.6330),
    'TLEMCEN': (34.8817, -1.3156),
    'ADRAR': (27.8766, -0.2898),
    'OUARGLA': (31.9454, 5.3267),
    'GHARDAIA': (32.4855, 3.6711),
    'BECHAR': (31.6295, -2.2269),
    'MASCARA': (35.4025, 0.1397),
    'MOSTAGANEM': (35.9339, 0.0881),
    'RELIZANE': (35.7397, 0.5589),
    'SIDI BEL ABBES': (35.1947, -0.6436),
    'SKIKDA': (36.8793, 6.9147),
    'GUELMA': (36.4619, 7.4314),
    'MILA': (36.4523, 6.2633),
    'B.B ARRERIDJ': (36.0732, 4.7592),
    'SOUK AHRAS': (36.2803, 7.9314),
    'KHENCHELA': (35.4314, 7.1411),
    'OUM EL BOUAGHI': (35.8667, 7.1097),
    'BATNA': (35.5547, 6.1744),
}

ZONE_COLORS = {
    'Zone 0': '#43a047',
    'Zone I': '#8bc34a',
    'Zone IIa': '#ffc107',
    'Zone IIb': '#ff9800',
    'Zone III': '#e53935',
}

def create_square_boundary(lat, lon, radius=0.3):
    """Create a square boundary around a point (approximates wilaya polygon)."""
    return [
        [lon - radius, lat - radius],
        [lon + radius, lat - radius],
        [lon + radius, lat + radius],
        [lon - radius, lat + radius],
        [lon - radius, lat - radius]  # Close polygon
    ]

def aggregate_by_wilaya(decisions_df):
    """Aggregate decisions, capital, PML by wilaya."""
    wilaya_stats = defaultdict(lambda: {
        'capital': 0,
        'pml': 0,
        'communes': set(),
        'decisions': defaultdict(int),
        'zone': None
    })
    
    for _, row in decisions_df.iterrows():
        wilaya = row['WILAYA'].strip()
        wilaya_stats[wilaya]['capital'] += float(row['capital_B'])
        wilaya_stats[wilaya]['pml'] += float(row['pml_B'])
        wilaya_stats[wilaya]['communes'].add(row['COMMUNE'])
        wilaya_stats[wilaya]['decisions'][row['DECISION']] += 1
        wilaya_stats[wilaya]['zone'] = row['ZONE_RPA']
    
    return wilaya_stats

def determine_dominant_decision(decisions_dict):
    """Return the most common decision for a wilaya."""
    if not decisions_dict:
        return 'ACCEPT'
    return max(decisions_dict, key=decisions_dict.get)

def generate_geojson(decisions_csv_path, output_geojson_path):
    """Generate GeoJSON from decision data."""
    print(f"📖 Reading {decisions_csv_path}...")
    df = pd.read_csv(decisions_csv_path)
    print(f"   Loaded {len(df)} communes across {df['WILAYA'].nunique()} wilayas")
    
    print("📊 Aggregating by wilaya...")
    wilaya_stats = aggregate_by_wilaya(df)
    
    features = []
    for wilaya, stats in wilaya_stats.items():
        if wilaya not in WILAYA_CENTERS:
            print(f"   ⚠️  {wilaya} not in coordinates dict, skipping")
            continue
        
        lat, lon = WILAYA_CENTERS[wilaya]
        zone = stats['zone'] or 'Zone IIa'
        dominant_decision = determine_dominant_decision(stats['decisions'])
        
        feature = {
            "type": "Feature",
            "properties": {
                "wilaya": wilaya,
                "zone": zone,
                "capital_B": round(stats['capital'], 2),
                "pml_B": round(stats['pml'], 2),
                "communes_count": len(stats['communes']),
                "decision_REJECT": stats['decisions'].get('REJECT', 0),
                "decision_ADJUST": stats['decisions'].get('ADJUST', 0),
                "decision_ACCEPT": stats['decisions'].get('ACCEPT', 0),
                "dominant_decision": dominant_decision,
                "color": ZONE_COLORS.get(zone, '#64748b'),
                "lat": lat,
                "lon": lon,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [create_square_boundary(lat, lon, radius=0.25)]
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    print(f"✅ Writing GeoJSON with {len(features)} features to {output_geojson_path}...")
    with open(output_geojson_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    
    print(f"✅ GeoJSON generated: {len(features)} wilayas")
    return output_geojson_path

if __name__ == "__main__":
    decisions_file = "decision_output_final.csv"
    output_file = "dashboard/data/algeria_wilayas.geojson"
    
    generate_geojson(decisions_file, output_file)
    print(f"\n✅ Ready for Leaflet.js map: {output_file}")
