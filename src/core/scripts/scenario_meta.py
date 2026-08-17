from __future__ import annotations

SCENARIO_RECORDS = [
    {"code":"SC0","short_label":"Baseline","family":"Baseline continuation","role":"reference pathway","overlay_file":None,"color":"black"},
    {"code":"SC1","short_label":"Demand smoothing","family":"Activity moderation","role":"primary evidence","overlay_file":"SC1_demand_smoothing.yaml","color":"#1f9eb7"},
    {"code":"SC2","short_label":"Access charging","family":"Pricing and regulation","role":"primary evidence","overlay_file":"SC2_access_charging.yaml","color":"#E15759"},
    {"code":"SC3","short_label":"Parking and curb management","family":"Pricing and regulation","role":"primary evidence","overlay_file":"SC3_parking_curb_management.yaml","color":"#4E79A7"},
    {"code":"SC4","short_label":"Fare support","family":"Public-transport conditions","role":"primary evidence","overlay_file":"SC4_fare_support.yaml","color":"#76B7B2"},
    {"code":"SC5","short_label":"Public-transport service improvement","family":"Public-transport conditions","role":"primary evidence","overlay_file":"SC5_public_transport_service_improvement.yaml","color":"#59A14F"},
    {"code":"SC6","short_label":"Local pollutant cleanup","family":"Local pollutant control","role":"supporting evidence","overlay_file":"SC6_local_pollutant_cleanup.yaml","color":"#EDC948"},
    {"code":"SC7","short_label":"Clean-fleet transition","family":"Fleet-energy transition","role":"supporting evidence","overlay_file":"SC7_clean_fleet_transition.yaml","color":"#B07AA1"},
    {"code":"SC8","short_label":"Balanced package","family":"Integrated policy packages","role":"solution package","overlay_file":"SC8_balanced_package.yaml","color":"#9C755F"},
    {"code":"SC9","short_label":"Access-led package","family":"Integrated policy packages","role":"solution package","overlay_file":"SC9_access_led_package.yaml","color":"#BAB0AC"},
    {"code":"SC10","short_label":"PT-first clean package","family":"Integrated policy packages","role":"solution package","overlay_file":"SC10_pt_first_clean_package.yaml","color":"#8E63CE"},
    {"code":"SC11","short_label":"Broad package","family":"Integrated policy packages","role":"solution package","overlay_file":"SC11_broad_package.yaml","color":"#FF9DA7"},
]

SCENARIO_CODES = [r["code"] for r in SCENARIO_RECORDS]
BASE_SCENARIO = "SC0"
SCENARIO_TO_COLOR = {r["code"]: r["color"] for r in SCENARIO_RECORDS}
SCENARIO_TO_OVERLAY = {r["code"]: r["overlay_file"] for r in SCENARIO_RECORDS if r["overlay_file"]}
SCENARIO_IDENTITY = {code: code for code in SCENARIO_CODES}
TECH_CODES = ["SC7"]
BEHAVIOR_CODES = ["SC1","SC2","SC3","SC4","SC5","SC6","SC8","SC9","SC10","SC11"]
CANDIDATE_CODES = [code for code in SCENARIO_CODES if code != BASE_SCENARIO]
DIAGNOSTIC_CODES = ["SC1","SC2","SC3","SC4","SC5","SC6","SC7"]
EXPLORATORY_CODES = []
