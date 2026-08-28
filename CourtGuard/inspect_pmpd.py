"""Quick PMPD database inspector — run after bootstrap to see all modules."""
from infrastructure.pmpd_repository import PMPDRepository

repo = PMPDRepository("pmpd_store.json")
db   = repo.load()

db.print_stats()

print("\n=== GLOBAL MODULE ===")
if db.global_module:
    gm = db.global_module
    print(f"Objective : {gm.objective}")
    print(f"Principles: {len(gm.principles)}")
    for p in gm.principles:
        print(f"  • {p}")
    print(f"\nDefinitions ({len(gm.definitions)}):")
    for term, defn in gm.definitions.items():
        print(f"  {term}: {defn}")
    print(f"\nEval Principles ({len(gm.general_eval_principles)}):")
    for i, ep in enumerate(gm.general_eval_principles, 1):
        print(f"  {i}. {ep}")

print("\n=== CATEGORY MODULES ===")
for cid in db.list_categories():
    cat = db.get_category(cid)
    print(f"\n[{cid}] {cat.name}")
    print(f"  Short def : {cat.short_definition}")
    print(f"  Rules     : {cat.general_rules[:150]}...")
    print(f"  Sub-cats  : {len(cat.sub_categories)}")
    for sub in cat.sub_categories:
        print(f"    {sub.get('id')} — {sub.get('name')}")
    print(f"  Exceptions: {len(cat.exceptions)}")
    for exc in cat.exceptions:
        print(f"    • {exc}")
    print(f"  One-shots : {len(cat.get_shadow_shots())} shadow, "
          f"{len(cat.get_live_shots(limit=999))} live")
    print(f"  Citations : {len(cat.citations)}")