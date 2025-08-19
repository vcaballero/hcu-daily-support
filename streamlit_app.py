import streamlit as st
import sqlite3
from datetime import datetime, date, timedelta, time
import pandas as pd
from typing import Optional, Tuple
import requests
import io
import urllib.parse

DB_PATH = 'tracker.db'

# ----------------------------
# Utilities & DB
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('PRAGMA foreign_keys = ON;')
    return conn


def init_db(conn):
    cur = conn.cursor()
    # Foods
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS foods (
               id INTEGER PRIMARY KEY,
               name TEXT NOT NULL,
               pack TEXT NOT NULL,
               protein_per_100g REAL NOT NULL,
               type TEXT CHECK(type IN ("raw", "processed")) NOT NULL DEFAULT "raw",
               barcode TEXT,
               UNIQUE(name, pack, COALESCE(barcode, ''))
           )'''
    )
    # Recipes
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS recipes (
               id INTEGER PRIMARY KEY,
               name TEXT NOT NULL UNIQUE,
               is_low_protein INTEGER NOT NULL DEFAULT 1,
               created_at TEXT NOT NULL
           )'''
    )
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS recipe_items (
               recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
               food_id INTEGER NOT NULL REFERENCES foods(id) ON DELETE RESTRICT,
               default_grams REAL NOT NULL DEFAULT 0,
               PRIMARY KEY(recipe_id, food_id)
           )'''
    )
    # Meals
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS meals (
               id INTEGER PRIMARY KEY,
               day TEXT NOT NULL,
               meal_type TEXT CHECK(meal_type IN (
                   "petit_dejeuner","dejeuner","gouter","diner"
               )) NOT NULL
           )'''
    )
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS meal_items (
               id INTEGER PRIMARY KEY,
               meal_id INTEGER NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
               food_id INTEGER NOT NULL REFERENCES foods(id) ON DELETE RESTRICT,
               grams REAL NOT NULL DEFAULT 0,
               from_recipe INTEGER NOT NULL DEFAULT 0,
               recipe_id INTEGER REFERENCES recipes(id) ON DELETE SET NULL
           )'''
    )
    # Treatments
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS treatments (
               id INTEGER PRIMARY KEY,
               name TEXT NOT NULL UNIQUE,
               periodicity TEXT CHECK(periodicity IN (
                   "1x/day","2x/day","3x/day","1x/week","2x/week","3x/week"
               )) NOT NULL,
               unit TEXT NOT NULL,
               quantity_per_intake REAL NOT NULL
           )'''
    )
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS treatment_logs (
               id INTEGER PRIMARY KEY,
               treatment_id INTEGER NOT NULL REFERENCES treatments(id) ON DELETE CASCADE,
               ts TEXT NOT NULL
           )'''
    )
    # Thresholds
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS thresholds (
               meal_type TEXT PRIMARY KEY,
               max_protein_g REAL NOT NULL
           )'''
    )
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS thresholds_daily (
               id INTEGER PRIMARY KEY CHECK(id = 1),
               max_protein_g REAL NOT NULL
           )'''
    )
    # Defaults if empty
    cur.execute('SELECT COUNT(*) FROM thresholds')
    if cur.fetchone()[0] == 0:
        cur.executemany('INSERT INTO thresholds(meal_type, max_protein_g) VALUES (?,?)', [
            ("petit_dejeuner", 2.0),
            ("gouter", 2.0),
            ("dejeuner", 4.0),
            ("diner", 4.0),
        ])
    cur.execute('SELECT COUNT(*) FROM thresholds_daily')
    if cur.fetchone()[0] == 0:
        cur.execute('INSERT INTO thresholds_daily(id, max_protein_g) VALUES (1, ?)', (10.0,))
    conn.commit()


conn = get_conn()
init_db(conn)

MEAL_LABELS = {
    "petit_dejeuner": "Petit déjeuner",
    "dejeuner": "Déjeuner",
    "gouter": "Goûter",
    "diner": "Dîner",
}

# ----------------------------
# Helper functions
# ----------------------------

def protein_for_food(grams: float, protein_per_100g: float) -> float:
    return round((grams * protein_per_100g) / 100.0, 3)


def get_foods_df() -> pd.DataFrame:
    return pd.read_sql_query('SELECT * FROM foods ORDER BY name, pack', conn)


def get_recipes_df() -> pd.DataFrame:
    df = pd.read_sql_query('SELECT * FROM recipes ORDER BY name', conn)
    return df


def get_recipe_items(recipe_id: int) -> pd.DataFrame:
    q = '''SELECT ri.food_id, f.name, f.pack, f.protein_per_100g, ri.default_grams
           FROM recipe_items ri JOIN foods f ON f.id = ri.food_id
           WHERE ri.recipe_id = ? ORDER BY f.name'''
    return pd.read_sql_query(q, conn, params=(recipe_id,))


def upsert_food(name: str, pack: str, protein: float, ftype: str, barcode: Optional[str]) -> Tuple[bool, Optional[int], Optional[str]]:
    try:
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO foods(name, pack, protein_per_100g, type, barcode) VALUES (?,?,?,?,?)',
            (name.strip(), pack.strip(), float(protein), ftype, barcode.strip() if barcode else None)
        )
        conn.commit()
        return True, cur.lastrowid, None
    except sqlite3.IntegrityError as e:
        return False, None, str(e)


def ensure_unique_recipe_name(name: str) -> str:
    # If name exists, append date suffix AAAAMMJJ as requested
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM recipes WHERE name = ?', (name,))
    exists = cur.fetchone()[0] > 0
    if exists:
        suffix = datetime.now().strftime('%Y%m%d')
        name2 = f"{name}_{suffix}"
        return name2
    return name


def add_recipe(name: str, is_low: bool) -> Tuple[bool, Optional[int], Optional[str], str]:
    try:
        name_final = ensure_unique_recipe_name(name.strip())
        cur = conn.cursor()
        cur.execute('INSERT INTO recipes(name, is_low_protein, created_at) VALUES (?,?,?)',
                    (name_final, 1 if is_low else 0, datetime.now().isoformat()))
        conn.commit()
        return True, cur.lastrowid, None, name_final
    except sqlite3.IntegrityError as e:
        return False, None, str(e), name


def add_recipe_item(recipe_id: int, food_id: int, default_grams: float):
    cur = conn.cursor()
    cur.execute('INSERT OR REPLACE INTO recipe_items(recipe_id, food_id, default_grams) VALUES (?,?,?)',
                (recipe_id, food_id, default_grams))
    conn.commit()


def get_or_create_meal(day_str: str, meal_type: str) -> int:
    cur = conn.cursor()
    cur.execute('SELECT id FROM meals WHERE day = ? AND meal_type = ?', (day_str, meal_type))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute('INSERT INTO meals(day, meal_type) VALUES (?,?)', (day_str, meal_type))
    conn.commit()
    return cur.lastrowid


def add_meal_item(meal_id: int, food_id: int, grams: float, from_recipe: bool=False, recipe_id: Optional[int]=None):
    cur = conn.cursor()
    cur.execute('INSERT INTO meal_items(meal_id, food_id, grams, from_recipe, recipe_id) VALUES (?,?,?,?,?)',
                (meal_id, food_id, grams, 1 if from_recipe else 0, recipe_id))
    conn.commit()


def remove_meal_item(item_id: int):
    conn.execute('DELETE FROM meal_items WHERE id = ?', (item_id,))
    conn.commit()


def meal_items_df(day_str: str, meal_type: str) -> pd.DataFrame:
    q = '''SELECT mi.id, mi.meal_id, mi.food_id, f.name, f.pack, f.protein_per_100g, mi.grams,
                  mi.from_recipe, mi.recipe_id
           FROM meal_items mi
           JOIN meals m ON m.id = mi.meal_id
           JOIN foods f ON f.id = mi.food_id
           WHERE m.day = ? AND m.meal_type = ?
           ORDER BY mi.id DESC'''
    return pd.read_sql_query(q, conn, params=(day_str, meal_type))


def meal_protein_total(day_str: str, meal_type: str) -> float:
    df = meal_items_df(day_str, meal_type)
    if df.empty:
        return 0.0
    return round((df['grams'] * df['protein_per_100g'] / 100).sum(), 3)


def day_protein_total(day_str: str) -> float:
    total = 0.0
    for mt in MEAL_LABELS.keys():
        total += meal_protein_total(day_str, mt)
    return round(total, 3)


def get_thresholds() -> Tuple[pd.DataFrame, float]:
    th = pd.read_sql_query('SELECT * FROM thresholds', conn).set_index('meal_type')
    daily = pd.read_sql_query('SELECT max_protein_g FROM thresholds_daily WHERE id = 1', conn)
    daily_max = float(daily.iloc[0,0]) if not daily.empty else 9999.0
    return th, daily_max


def save_thresholds(th_df: pd.DataFrame, daily_max: float):
    cur = conn.cursor()
    for mt, row in th_df.iterrows():
        cur.execute('UPDATE thresholds SET max_protein_g = ? WHERE meal_type = ?', (float(row['max_protein_g']), mt))
    cur.execute('INSERT OR REPLACE INTO thresholds_daily(id, max_protein_g) VALUES (1, ?)', (float(daily_max),))
    conn.commit()


def add_treatment(name: str, periodicity: str, unit: str, qty: float):
    cur = conn.cursor()
    cur.execute('INSERT OR REPLACE INTO treatments(name, periodicity, unit, quantity_per_intake) VALUES (?,?,?,?)',
                (name.strip(), periodicity, unit.strip(), float(qty)))
    conn.commit()


def treatments_df() -> pd.DataFrame:
    return pd.read_sql_query('SELECT * FROM treatments ORDER BY name', conn)


def log_treatment_intake(treatment_id: int, when: Optional[datetime] = None):
    ts = (when or datetime.now()).isoformat()
    conn.execute('INSERT INTO treatment_logs(treatment_id, ts) VALUES (?,?)', (treatment_id, ts))
    conn.commit()


def treatment_logs_between(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    q = 'SELECT * FROM treatment_logs WHERE ts BETWEEN ? AND ?'
    return pd.read_sql_query(q, conn, params=(start_dt.isoformat(), end_dt.isoformat()))


def expected_intakes_for_periodicity(periodicity: str, day: date) -> int:
    if periodicity == '1x/day':
        return 1
    if periodicity == '2x/day':
        return 2
    if periodicity == '3x/day':
        return 3
    # weekly cases: we'll handle on a weekly normalization
    if periodicity in ('1x/week','2x/week','3x/week'):
        return 0
    return 0


def weekly_expected(periodicity: str) -> int:
    return {
        '1x/week': 1,
        '2x/week': 2,
        '3x/week': 3,
    }.get(periodicity, 0)

# ---- Airtable & Export helpers ----

def airtable_fetch_all(api_key: str, base_id: str, table_name: str, view: Optional[str] = None, max_records: int = 10000):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://api.airtable.com/v0/{base_id}/{urllib.parse.quote(table_name)}"
    params = {}
    if view:
        params["view"] = view
    out = []
    offset = None
    fetched = 0
    while True:
        p = params.copy()
        if offset:
            p["offset"] = offset
        resp = requests.get(url, headers=headers, params=p, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        recs = data.get("records", [])
        out.extend(recs)
        fetched += len(recs)
        offset = data.get("offset")
        if not offset or fetched >= max_records:
            break
    return out


def import_foods_from_airtable(api_key: str, base_id: str, table_name: str,
                               field_name: str, field_pack: str, field_protein: str,
                               field_type: Optional[str] = None, field_barcode: Optional[str] = None) -> Tuple[int,int,int]:
    records = airtable_fetch_all(api_key, base_id, table_name)
    added = skipped = errors = 0
    for r in records:
        fields = r.get('fields', {})
        try:
            name = str(fields.get(field_name, '')).strip()
            pack = str(fields.get(field_pack, '')).strip()
            if not name or not pack:
                skipped += 1
                continue
            protein_raw = fields.get(field_protein, 0)
            try:
                protein = float(protein_raw)
            except (ValueError, TypeError):
                skipped += 1
                continue
            ftype_val = (fields.get(field_type) if field_type else 'raw') or 'raw'
            ftype = 'processed' if str(ftype_val).lower().startswith('trans') else ('raw' if str(ftype_val).lower().startswith('non') else str(ftype_val).lower())
            if ftype not in ('raw','processed'):
                ftype = 'raw'
            barcode = str(fields.get(field_barcode, '')).strip() if field_barcode else None
            ok, _, _ = upsert_food(name, pack, protein, ftype, barcode or None)
            if ok:
                added += 1
            else:
                skipped += 1
        except Exception:
            errors += 1
    return added, skipped, errors


def df_meals_between(start: date, end: date) -> pd.DataFrame:
    q = '''SELECT m.day, m.meal_type, mi.id as item_id, f.name as food_name, f.pack, f.protein_per_100g,
                  mi.grams, (mi.grams * f.protein_per_100g / 100.0) as protein_g
           FROM meal_items mi
           JOIN meals m ON m.id = mi.meal_id
           JOIN foods f ON f.id = mi.food_id
           WHERE date(m.day) BETWEEN date(?) AND date(?)
           ORDER BY m.day, m.meal_type, mi.id'''
    return pd.read_sql_query(q, conn, params=(start.isoformat(), end.isoformat()))


def df_recipe_items_all() -> pd.DataFrame:
    q = '''SELECT r.name as recipe_name, r.is_low_protein, f.name as food_name, f.pack, f.protein_per_100g, ri.default_grams,
                  (ri.default_grams * f.protein_per_100g / 100.0) as protein_g_default
           FROM recipe_items ri
           JOIN recipes r ON r.id = ri.recipe_id
           JOIN foods f ON f.id = ri.food_id
           ORDER BY r.name, f.name'''
    return pd.read_sql_query(q, conn)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# ----------------------------
# UI Components
# ----------------------------
st.set_page_config(page_title="HCU – Suivi protéines & traitements", layout="wide")
st.title("Suivi protéines & traitements – HCU")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Aller à", (
        "Suivi journalier",
        "Gestion base de données aliments",
        "Gestion base de données recettes",
        "Gestion base de données traitements",
        "Import / Export",
        "Paramètres / Seuils",
    ))
    st.markdown("---")
    st.caption("Astuce: sur smartphone, vous pouvez ajouter des aliments transformés avec leur code barre (saisie manuelle pour l'instant).")

# ----------------------------
# Page: Suivi journalier
# ----------------------------
if page == "Suivi journalier":
    colL, colR = st.columns([2,1])
    with colL:
        st.subheader("Vue d'ensemble")
        today = st.date_input("Jour", value=date.today())
        day_str = today.isoformat()

        # Chart on last 14 days
        lookback = 14
        days = [date.today() - timedelta(days=i) for i in range(lookback)][::-1]
        data_rows = []
        for d in days:
            ds = d.isoformat()
            totals = {mt: meal_protein_total(ds, mt) for mt in MEAL_LABELS.keys()}
            row = {"date": ds, **totals, "total": sum(totals.values())}
            data_rows.append(row)
        chart_df = pd.DataFrame(data_rows)
        st.line_chart(chart_df.set_index('date')[['petit_dejeuner','dejeuner','gouter','diner','total']])

    with colR:
        st.subheader("Traitements – progression (7 jours)")
        tdf = treatments_df()
        if tdf.empty:
            st.info("Aucun traitement défini pour l'instant.")
        else:
            start = datetime.combine(date.today() - timedelta(days=6), time.min)
            end = datetime.combine(date.today(), time.max)
            logs = treatment_logs_between(start, end)
            # Build compliance table
            records = []
            for _, t in tdf.iterrows():
                total_logged = logs[logs['treatment_id'] == t['id']].shape[0]
                weekly_target = weekly_expected(t['periodicity'])
                if weekly_target > 0:
                    expected = weekly_target
                else:
                    per_day = expected_intakes_for_periodicity(t['periodicity'], date.today())
                    expected = per_day * 7
                rate = (total_logged / expected * 100) if expected > 0 else (100 if total_logged > 0 else 0)
                records.append({
                    'Traitement': t['name'],
                    'Périodicité': t['periodicity'],
                    'Pris (7j)': total_logged,
                    'Attendu (7j)': expected,
                    'Adhérence %': round(rate, 1),
                })
            comp_df = pd.DataFrame(records)
            st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Enregistrer un repas")
    meal_type = st.selectbox("Type de repas", list(MEAL_LABELS.keys()), format_func=lambda k: MEAL_LABELS[k])
    meal_id = get_or_create_meal(day_str, meal_type)

    tab_add_recipe, tab_add_food, tab_log_treat = st.tabs([
        "Ajouter depuis une recette",
        "Ajouter un aliment",
        "Valider une prise de médicament",
    ])

    with tab_add_recipe:
        r_df = get_recipes_df()
        if r_df.empty:
            st.info("Aucune recette pour l'instant. Ajoutez-en dans l'onglet Recettes ou directement ci-dessous.")
        recipe_choice = st.selectbox("Recette", [None] + r_df['name'].tolist())
        if recipe_choice:
            r_id = int(r_df[r_df['name'] == recipe_choice]['id'].iloc[0])
            items = get_recipe_items(r_id)
            if items.empty:
                st.warning("Cette recette n'a pas encore d'ingrédients.")
            else:
                st.write("Renseignez les grammes servis par ingrédient :")
                grams_inputs = {}
                for idx, row in items.iterrows():
                    col1, col2, col3, col4 = st.columns([2,1,1,1])
                    with col1:
                        st.caption(f"{row['name']} ({row['pack']}) – {row['protein_per_100g']} g/100g")
                    with col2:
                        grams_inputs[row['food_id']] = st.number_input(
                            f"Grammes – {row['name']}", min_value=0.0, value=float(row['default_grams']), step=5.0, key=f"grams_{r_id}_{row['food_id']}"
                        )
                    with col3:
                        st.write("")
                        st.write("")
                        st.write(f"= {protein_for_food(grams_inputs[row['food_id']], row['protein_per_100g'])} g prot.")
                    with col4:
                        st.write("")
                if st.button("Ajouter au repas", key="add_recipe_to_meal"):
                    for fid, g in grams_inputs.items():
                        if g and g > 0:
                            add_meal_item(meal_id, int(fid), float(g), from_recipe=True, recipe_id=r_id)
                    st.success("Ingrédients ajoutés au repas.")
                    st.experimental_rerun()
        st.markdown("---")
        st.caption("Créer une recette à la volée")
        with st.form("quick_add_recipe"):
            q_name = st.text_input("Nom de la recette")
            q_low = st.checkbox("Hypoprotéinée", value=True)
            submitted = st.form_submit_button("Créer")
            if submitted:
                ok, rid, err, final_name = add_recipe(q_name, q_low)
                if ok:
                    st.success(f"Recette créée: {final_name}")
                else:
                    st.error(f"Erreur création recette: {err}")

    with tab_add_food:
        foods = get_foods_df()
        if foods.empty:
            st.info("Aucun aliment. Ajoutez-en ci-dessous.")
        # Select food
        if not foods.empty:
            food_labels = foods.apply(lambda r: f"{r['name']} – {r['pack']} ({r['protein_per_100g']} g/100g)", axis=1).tolist()
            food_map = {label: fid for label, fid in zip(food_labels, foods['id'])}
            choice = st.selectbox("Aliment", [None] + food_labels)
            if choice:
                fid = food_map[choice]
                row = foods[foods['id'] == fid].iloc[0]
                grams = st.number_input("Quantité servie (g)", min_value=0.0, value=0.0, step=5.0)
                st.write(f"Teneur protéines: {row['protein_per_100g']} g/100g")
                st.write(f"Protéines servies = {protein_for_food(grams, row['protein_per_100g'])} g")
                if st.button("Ajouter au repas", key="add_food_to_meal") and grams > 0:
                    add_meal_item(meal_id, int(fid), float(grams), from_recipe=False)
                    st.success("Aliment ajouté au repas.")
                    st.experimental_rerun()

        st.markdown("---")
        st.caption("Ajouter un aliment à la volée")
        with st.form("quick_add_food"):
            name = st.text_input("Nom")
            pack = st.text_input("Conditionnement (cru, cuit, boîte, surgelé, ...)")
            protein = st.number_input("Taux de protéines (g/100g)", min_value=0.0, step=0.1)
            ftype = st.selectbox("Type", ("raw","processed"), format_func=lambda x: "non transformé" if x=="raw" else "transformé")
            barcode = st.text_input("Code barre (optionnel)")
            submitted = st.form_submit_button("Enregistrer l'aliment")
            if submitted:
                ok, fid, err = upsert_food(name, pack, protein, ftype, barcode or None)
                if ok:
                    st.success("Aliment enregistré.")
                else:
                    st.error(f"Échec (doublon probable nom+conditionnement+code barre): {err}")

    with tab_log_treat:
        tdf = treatments_df()
        if tdf.empty:
            st.info("Définissez d'abord vos traitements dans le menu dédié.")
        else:
            t_choice = st.selectbox("Médicament", tdf['name'].tolist())
            when = st.datetime_input("Date/heure de prise", value=datetime.now())
            if st.button("Valider la prise"):
                tid = int(tdf[tdf['name'] == t_choice]['id'].iloc[0])
                log_treatment_intake(tid, when)
                st.success("Prise enregistrée.")

    st.markdown("---")
    st.subheader("Bilan du repas / du jour")
    # Per-meal total
    m_total = meal_protein_total(day_str, meal_type)
    th_df, daily_max = get_thresholds()
    meal_cap = float(th_df.loc[meal_type, 'max_protein_g']) if meal_type in th_df.index else 9999.0
    st.metric(label=f"{MEAL_LABELS[meal_type]} – Protéines servies", value=f"{m_total} g", delta=f"Seuil {meal_cap} g")

    # Show current meal items table
    df_items = meal_items_df(day_str, meal_type)
    if not df_items.empty:
        df_view = df_items.copy()
        df_view['Protéines (g)'] = (df_view['grams'] * df_view['protein_per_100g'] / 100).round(3)
        df_view = df_view[['id','name','pack','grams','protein_per_100g','Protéines (g)']]
        st.dataframe(df_view.rename(columns={'name':'Aliment','pack':'Cond.','grams':'Grammes','protein_per_100g':'Prot/100g'}), use_container_width=True)
        rem_id = st.selectbox("Retirer un élément", [None] + df_items['id'].tolist())
        if rem_id and st.button("Retirer"):
            remove_meal_item(int(rem_id))
            st.experimental_rerun()

    # Daily total vs threshold
    d_total = day_protein_total(day_str)
    st.metric(label="Total jour – protéines servies", value=f"{d_total} g", delta=f"Seuil {daily_max} g")

# ----------------------------
# Page: Gestion base de données aliments
# ----------------------------
elif page == "Gestion base de données aliments":
    st.subheader("Aliments non transformés")
    with st.expander("Ajouter / modifier"):
        with st.form("add_raw_food"):
            name = st.text_input("Nom (ex: épinards)")
            pack = st.text_input("Conditionnement (cru, cuit, surgelé, ...)")
            protein = st.number_input("Taux de protéines (g/100g)", min_value=0.0, step=0.1)
            submitted = st.form_submit_button("Enregistrer")
            if submitted:
                ok, fid, err = upsert_food(name, pack, protein, 'raw', None)
                if ok:
                    st.success("Aliment enregistré.")
                else:
                    st.error(f"Échec (doublon probable nom+conditionnement): {err}")

    df = get_foods_df()
    raw_df = df[df['type'] == 'raw']
    if not raw_df.empty:
        st.dataframe(raw_df.rename(columns={'name':'Nom','pack':'Conditionnement','protein_per_100g':'Prot/100g'}), use_container_width=True)
    else:
        st.info("Aucun aliment non transformé.")

    st.markdown("---")
    st.subheader("Aliments transformés (avec code barre)")
    with st.expander("Ajouter / modifier"):
        with st.form("add_proc_food"):
            name = st.text_input("Nom (ex: frites surgelées)", key="p_name")
            pack = st.text_input("Conditionnement (boîte, sachet, ...)", key="p_pack")
            protein = st.number_input("Taux de protéines (g/100g)", min_value=0.0, step=0.1, key="p_prot")
            barcode = st.text_input("Code barre (EAN)", key="p_barcode")
            submitted = st.form_submit_button("Enregistrer")
            if submitted:
                ok, fid, err = upsert_food(name, pack, protein, 'processed', barcode or None)
                if ok:
                    st.success("Aliment enregistré.")
                else:
                    st.error(f"Échec (doublon probable nom+conditionnement+code barre): {err}")

    proc_df = df[df['type'] == 'processed']
    if not proc_df.empty:
        st.dataframe(proc_df.rename(columns={'name':'Nom','pack':'Conditionnement','protein_per_100g':'Prot/100g','barcode':'Code barre'}), use_container_width=True)
    else:
        st.info("Aucun aliment transformé.")

# ----------------------------
# Page: Gestion base de données recettes
# ----------------------------
elif page == "Gestion base de données recettes":
    st.subheader("Recettes")

    with st.expander("Créer une recette"):
        with st.form("create_recipe"):
            name = st.text_input("Nom de la recette")
            is_low = st.checkbox("Hypoprotéinée", value=True)
            submitted = st.form_submit_button("Créer")
            if submitted:
                ok, rid, err, final_name = add_recipe(name, is_low)
                if ok:
                    st.success(f"Recette créée: {final_name}")
                else:
                    st.error(f"Erreur: {err}")

    r_df = get_recipes_df()
    if r_df.empty:
        st.info("Aucune recette enregistrée.")
    else:
        choice = st.selectbox("Sélectionner une recette", r_df['name'].tolist())
        rid = int(r_df[r_df['name'] == choice]['id'].iloc[0])
        st.caption("Ajouter des ingrédients (issus de la base aliments) et définir des grammes par défaut (peuvent être modifiés lors du service)")
        foods = get_foods_df()
        if foods.empty:
            st.warning("Ajoutez d'abord des aliments dans la base.")
        else:
            food_labels = foods.apply(lambda r: f"{r['name']} – {r['pack']} ({r['protein_per_100g']} g/100g)", axis=1).tolist()
            food_map = {label: fid for label, fid in zip(food_labels, foods['id'])}
            with st.form("add_item_to_recipe"):
                f_choice = st.selectbox("Aliment", food_labels)
                default_g = st.number_input("Grammes par défaut", min_value=0.0, step=5.0)
                submitted = st.form_submit_button("Ajouter à la recette")
                if submitted:
                    add_recipe_item(rid, int(food_map[f_choice]), float(default_g))
                    st.success("Ingrédient ajouté.")
        items = get_recipe_items(rid)
        if not items.empty:
            items = items.copy()
            items['Prot/100g'] = items['protein_per_100g']
            items['Prot (par défaut)'] = (items['default_grams'] * items['protein_per_100g'] / 100).round(3)
            items = items[['name','pack','default_grams','Prot/100g','Prot (par défaut)']]
            st.dataframe(items.rename(columns={'name':'Aliment','pack':'Cond.','default_grams':'Grammes défaut'}), use_container_width=True)

# ----------------------------
# Page: Gestion base de données traitements
# ----------------------------
elif page == "Gestion base de données traitements":
    st.subheader("Traitements")

    with st.expander("Ajouter / modifier un traitement"):
        with st.form("add_treatment"):
            name = st.text_input("Nom du médicament")
            periodicity = st.selectbox("Périodicité", ("1x/day","2x/day","3x/day","1x/week","2x/week","3x/week"),
                                      format_func=lambda x: x.replace('x','×').replace('/day','/jour').replace('/week','/semaine'))
            unit = st.text_input("Unité (ex: cachets, ampoules, g)")
            qty = st.number_input("Quantité à chaque prise", min_value=0.0, step=0.5)
            submitted = st.form_submit_button("Enregistrer")
            if submitted:
                add_treatment(name, periodicity, unit, qty)
                st.success("Traitement enregistré.")

    tdf = treatments_df()
    if not tdf.empty:
        st.dataframe(tdf.rename(columns={'name':'Nom','periodicity':'Périodicité','unit':'Unité','quantity_per_intake':'Qté / prise'}), use_container_width=True)
    else:
        st.info("Aucun traitement défini.")

# ----------------------------
# Page: Import / Export
# ----------------------------
elif page == "Import / Export":
    st.subheader("Import Airtable → Aliments")
    st.caption("Renseignez votre clé API, Base ID et nom de table. Mappez les champs si besoin.")

    with st.form("airtable_conf"):
        c1, c2 = st.columns(2)
        api_key = c1.text_input("API Key (Airtable)", type="password")
        base_id = c2.text_input("Base ID")
        tname = st.text_input("Nom de la table", value="Suivi Aliments")

        st.markdown("**Mapping des champs**")
        f_name = st.text_input("Champ Nom", value="name")
        f_pack = st.text_input("Champ Conditionnement", value="pack")
        f_prot = st.text_input("Champ Protéines/100g", value="protein_per_100g")
        f_type = st.text_input("Champ Type (raw/processed) (optionnel)", value="type")
        f_bar  = st.text_input("Champ Code barre (optionnel)", value="barcode")

        submitted = st.form_submit_button("Importer les aliments")
        if submitted:
            if not api_key or not base_id or not tname:
                st.error("Renseignez clé API, Base ID et nom de table.")
            else:
                try:
                    added, skipped, errors = import_foods_from_airtable(
                        api_key, base_id, tname, f_name, f_pack, f_prot, f_type, f_bar
                    )
                    st.success(f"Import terminé – ajoutés: {added}, ignorés/duplicats: {skipped}, erreurs: {errors}")
                except Exception as e:
                    st.error(f"Erreur import: {e}")

    st.markdown("---")
    st.subheader("Export CSV")
    dfF  = get_foods_df()
    dfR  = get_recipes_df()
    dfRI = df_recipe_items_all()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Télécharger aliments.csv", to_csv_bytes(dfF), file_name="aliments.csv", mime="text/csv")
    with c2:
        st.download_button("Télécharger recettes.csv", to_csv_bytes(dfR), file_name="recettes.csv", mime="text/csv")
    with c3:
        st.download_button("Télécharger recettes_items.csv", to_csv_bytes(dfRI), file_name="recettes_items.csv", mime="text/csv")

    st.caption("Export des repas sur une période :")
    d1, d2 = st.columns(2)
    start_d = d1.date_input("Du", value=date.today() - timedelta(days=30))
    end_d   = d2.date_input("Au",  value=date.today())
    if start_d > end_d:
        st.error("La date de début doit précéder la date de fin.")
    else:
        meals_df = df_meals_between(start_d, end_d)
        st.download_button("Télécharger repas.csv", to_csv_bytes(meals_df), file_name="repas.csv", mime="text/csv")

# ----------------------------
# Page: Paramètres / Seuils
# ----------------------------
else:
    st.subheader("Seuils protidiques")
    th_df, daily_max = get_thresholds()
    edit = th_df.reset_index().rename(columns={'meal_type':'Repas','max_protein_g':'Seuil (g)'})
    st.caption("Par défaut: petit-déjeuner/goûter = 2 g, déjeuner/dîner = 4 g. Modifiez si besoin.")
    new_daily = st.number_input("Seuil total journalier (g)", min_value=0.0, step=0.5, value=float(daily_max))
    st.dataframe(edit, use_container_width=True)

    with st.form("save_thresholds_form"):
        c1, c2, c3, c4 = st.columns(4)
        v1 = c1.number_input("Petit déjeuner (g)", min_value=0.0, step=0.5, value=float(th_df.loc['petit_dejeuner','max_protein_g']))
        v2 = c2.number_input("Déjeuner (g)", min_value=0.0, step=0.5, value=float(th_df.loc['dejeuner','max_protein_g']))
        v3 = c3.number_input("Goûter (g)", min_value=0.0, step=0.5, value=float(th_df.loc['gouter','max_protein_g']))
        v4 = c4.number_input("Dîner (g)", min_value=0.0, step=0.5, value=float(th_df.loc['diner','max_protein_g']))
        submitted = st.form_submit_button("Enregistrer les seuils")
        if submitted:
            new_df = pd.DataFrame({'max_protein_g':[v1,v2,v3,v4]}, index=['petit_dejeuner','dejeuner','gouter','diner'])
            save_thresholds(new_df, new_daily)
            st.success("Seuils enregistrés.")

st.markdown("\n\n——\n*Construit pour suivre précisément les protéines et les traitements au quotidien (homocystinurie).*")
