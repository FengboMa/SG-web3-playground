# Introduction page
import streamlit as st
import psycopg2
import pandas as pd
import pycountry
import datetime
import requests
import os
import uuid

# ---- MUST COME FIRST ----
st.set_page_config(page_title="SpectraGuru Auth Test")

# ---- HEADER / IMAGE ----
st.image(r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v14\element\Application header picture-3.png")

# ---- BASIC PAGE SETUP ----
st.session_state.log_file_path = r"C:\Users\zhaoy_admin\Desktop\OneDrive - University of Georgia\Research Group\Projects\2024-Redwan & Henry & Jiaheng-Spectra Analysis Software\spectraApp_v14\element\user_count.txt"

st.write("# SpectraGuru - A Spectra Analysis Application")
st.info("SpectraGuru Database registration test page")

st.title("🔬 SpectraGuru Database Gateway")


st.session_state.connection = False

with st.form("DB_login"):
    st.write("Log in to DB")
    st.session_state.user = st.text_input("User")
    st.session_state.passkey = st.text_input("Passkey")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Log in")
    if submitted:
        try:
            conn = psycopg2.connect(
                dbname="SpectraGuruDB",
                user=st.session_state.user,
                password=st.session_state.passkey,
                host="localhost",  # Use "localhost" for local database
                port="5432"  # Default PostgreSQL port
            )
            cur = conn.cursor()
            
            st.write("Connection to the database was successful.")
            st.session_state.connection = True

        except:
            # If an error occurs, print the error
            st.warning(f"The error occurred")

#########################
# Database search feature test

st.divider()
def get_db_connection():
    return psycopg2.connect(
        dbname="SpectraGuruDB",
        user=st.session_state.user,
        password=st.session_state.passkey,
        host="localhost",
        port="5432"
    )

def to_sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime.date):
        return f"'{value.isoformat()}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"

def build_spectrum_dump_sql(
    dump_id,
    batch_standard_id,
    spectrum_standard_id,
    spectrum_data_standard_start_id,
    spectrum_name,
    spectrum_df
):
    sql_lines = [
        "-- SpectraGuru SQL dump",
        f"-- dump_id: {dump_id}",
        f"-- generated_at_utc: {datetime.datetime.utcnow().isoformat()}",
        "BEGIN;",
        (
            "INSERT INTO spectrum_standard "
            "(spectrum_standard_id, spectrum_name, batch_standard_id) VALUES "
            f"({to_sql_literal(spectrum_standard_id)}, {to_sql_literal(spectrum_name)}, {to_sql_literal(batch_standard_id)});"
        ),
    ]

    next_data_id = spectrum_data_standard_start_id
    for _, row in spectrum_df.iterrows():
        sql_lines.append(
            "INSERT INTO spectrum_data_standard "
            "(spectrum_data_standard_id, spectrum_standard_id, wavenumber, intensity) VALUES "
            f"({to_sql_literal(next_data_id)}, {to_sql_literal(spectrum_standard_id)}, {to_sql_literal(row[0])}, {to_sql_literal(row[1])});"
        )
        next_data_id += 1

    sql_lines.append("COMMIT;")
    return "\n".join(sql_lines)

def save_sql_dump_and_stage(sql_text, dump_id, source_filename):
    sql_dump_dir = "sql_dumps"
    cloud_stage_dir = "cloud_stage"
    os.makedirs(sql_dump_dir, exist_ok=True)
    os.makedirs(cloud_stage_dir, exist_ok=True)

    safe_filename = os.path.basename(source_filename).replace(" ", "_")
    dump_filename = f"{dump_id}_{safe_filename}.sql"
    dump_path = os.path.join(sql_dump_dir, dump_filename)
    with open(dump_path, "w", encoding="utf-8") as f:
        f.write(sql_text)

    manifest_path = os.path.join(cloud_stage_dir, "pending_uploads_manifest.tsv")
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.datetime.utcnow().isoformat()}\t{dump_id}\t{dump_path}\tPENDING_UPLOAD\n"
        )

    return dump_path, manifest_path

def compute_relevance(row, keywords):
    return sum(
        any(str(keyword).lower() in str(cell).lower() for cell in row)
        for keyword in keywords
    )

# Search Function
def search_database(search_term, data_type_filter="Both"):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        search_pattern = f"%{search_term}%"

        # Query 1 (Search in Raw Data)
        query1 = """
        SELECT 
            u.user_id AS user_id,
            u.name AS user_name,
            u.location AS user_location,
            u.institution AS user_institution,
            p.project_id AS project_id,
            p.project_name AS project_name,
            p.start_date AS project_start_date,
            p.source AS project_source,
            db.batch_id AS batch_id,  -- Keep consistent column names
            db.upload_date AS batch_upload_date,
            db.analyte_name AS batch_analyte_name,
            db.buffer_solution AS batch_buffer_solution,
            db.instrument_details AS batch_instrument_details,
            db.wavelength AS batch_wavelength,
            db.power AS batch_power,
            db.concentration AS batch_concentration,
            db.concentration_units AS batch_concentration_units,
            db.accumulation_time AS batch_accumulation_time,
            db.experimental_procedure AS batch_experimental_procedure,
            db.substrate_type AS batch_substrate_type,
            db.substrate_material AS batch_substrate_material,
            db.preparation_conditions AS batch_preparation_conditions,
            db.data_type AS batch_data_type,
            db.notes AS batch_notes
        FROM
            "user" u
        JOIN
            project_user pu ON u.user_id = pu.user_id
        JOIN
            "project" p ON p.project_id = pu.project_id
        JOIN
            project_batch pb ON p.project_id = pb.project_id
        JOIN
            "databatch" db ON db.batch_id = pb.batch_id
        WHERE 
            COALESCE(u.name, '') ILIKE %s OR 
            COALESCE(u.location, '') ILIKE %s OR 
            COALESCE(u.institution, '') ILIKE %s OR 
            COALESCE(p.project_name, '') ILIKE %s OR
            COALESCE(p.source, '') ILIKE %s OR 
            COALESCE(db.analyte_name, '') ILIKE %s OR
            COALESCE(db.buffer_solution, '') ILIKE %s OR
            COALESCE(db.instrument_details, '') ILIKE %s OR
            COALESCE(db.experimental_procedure, '') ILIKE %s OR
            COALESCE(db.substrate_type, '') ILIKE %s OR
            COALESCE(db.substrate_material, '') ILIKE %s OR
            COALESCE(db.preparation_conditions, '') ILIKE %s OR
            COALESCE(db.data_type, '') ILIKE %s OR
            COALESCE(db.notes, '') ILIKE %s;
        """

        # Query 2 (Search in Standard Data)
        query2 = """
        SELECT 
            u.user_id AS user_id,
            u.name AS user_name,
            u.location AS user_location,
            u.institution AS user_institution,
            p.project_id AS project_id,
            p.project_name AS project_name,
            p.start_date AS project_start_date,
            p.source AS project_source,
            db.batch_standard_id AS batch_id,  -- Keep column names same as Query 1
            db.upload_date AS batch_upload_date,
            db.analyte_name AS batch_analyte_name,
            db.buffer_solution AS batch_buffer_solution,
            db.instrument_details AS batch_instrument_details,
            db.wavelength AS batch_wavelength,
            db.power AS batch_power,
            db.concentration AS batch_concentration,
            db.concentration_units AS batch_concentration_units,
            db.accumulation_time AS batch_accumulation_time,
            db.experimental_procedure AS batch_experimental_procedure,
            db.substrate_type AS batch_substrate_type,
            db.substrate_material AS batch_substrate_material,
            db.preparation_conditions AS batch_preparation_conditions,
            db.data_type AS batch_data_type,
            db.notes AS batch_notes
        FROM
            "user" u
        JOIN
            project_user pu ON u.user_id = pu.user_id
        JOIN
            "project" p ON p.project_id = pu.project_id
        JOIN
            project_batch_standard pb ON p.project_id = pb.project_id
        JOIN
            "databatch_standard" db ON db.batch_standard_id = pb.batch_standard_id
        WHERE 
            COALESCE(u.name, '') ILIKE %s OR 
            COALESCE(u.location, '') ILIKE %s OR 
            COALESCE(u.institution, '') ILIKE %s OR 
            COALESCE(p.project_name, '') ILIKE %s OR
            COALESCE(p.source, '') ILIKE %s OR 
            COALESCE(db.analyte_name, '') ILIKE %s OR
            COALESCE(db.buffer_solution, '') ILIKE %s OR
            COALESCE(db.instrument_details, '') ILIKE %s OR
            COALESCE(db.experimental_procedure, '') ILIKE %s OR
            COALESCE(db.substrate_type, '') ILIKE %s OR
            COALESCE(db.substrate_material, '') ILIKE %s OR
            COALESCE(db.preparation_conditions, '') ILIKE %s OR
            COALESCE(db.data_type, '') ILIKE %s OR
            COALESCE(db.notes, '') ILIKE %s;
        """
        results = []

        keywords = search_term.strip().split()
        if not keywords:
            return pd.DataFrame()

        # Get full OR pattern
        like_patterns = [f"%{k}%" for k in keywords]

        # Set of results
        results = []

        for pattern in like_patterns:
            params = (pattern,) * 14

            if data_type_filter in ["Both", "Raw Data Only"]:
                cur.execute(query1, params)
                rows1 = cur.fetchall()
                columns1 = [desc[0] for desc in cur.description]
                df1 = pd.DataFrame(rows1, columns=columns1)
                results.append(df1)

            if data_type_filter in ["Both", "Standard Data Only"]:
                cur.execute(query2, params)
                rows2 = cur.fetchall()
                columns2 = [desc[0] for desc in cur.description]
                df2 = pd.DataFrame(rows2, columns=columns2)
                results.append(df2)

        if results:
            result_df = pd.concat(results, ignore_index=True).drop_duplicates()

            # Add a relevance score column
            result_df["relevance"] = result_df.apply(lambda row: compute_relevance(row, keywords), axis=1)

            # Sort by relevance descending
            result_df = result_df.sort_values(by="relevance", ascending=False)
        else:
            result_df = pd.DataFrame()

        
        return result_df
        # # Execute Query 1
        # cur.execute(query1, (search_pattern,) * 14)  # Pass correct number of parameters
        # rows1 = cur.fetchall()
        # columns1 = [desc[0] for desc in cur.description]
        # df1 = pd.DataFrame(rows1, columns=columns1)

        # # Execute Query 2
        # cur.execute(query2, (search_pattern,) * 14)  # Pass correct number of parameters
        # rows2 = cur.fetchall()
        # columns2 = [desc[0] for desc in cur.description]
        # df2 = pd.DataFrame(rows2, columns=columns2)

        # # Merge results into one DataFrame
        # result_df = pd.concat([df1, df2], ignore_index=True)

        # conn.close()
        # return result_df
    except Exception as e:
        conn.close()
        st.error(f"Error fetching search results: {e}")
        return pd.DataFrame()

st.write("## Database Search")

advanced_search = st.checkbox("Advanced Search")

# If advanced search is selected, show additional filters
if advanced_search:
    data_type_filter = st.radio(
        "Select Data Type to Search:",
        options=["Both", "Raw Data Only", "Standard Data Only"],
        index=0,
        horizontal=True
    )
else:
    data_type_filter = "Both"

search_query = st.text_input("Enter a keyword to search in the database:", "")



if search_query:
    results = search_database(search_query, data_type_filter)
    if not results.empty:
        st.write("### Search Results")
        
        # Step 1: Add 'select' column (default to False)
        results['Select'] = False

        # Step 2: Reorder columns - Move 'select' to the very front
        columns_order = ['Select', 'batch_id', 'batch_analyte_name', 'batch_data_type'] + [col for col in results.columns if col not in ['Select', 'batch_id', 'batch_analyte_name', 'batch_data_type']]
        results = results[columns_order]

        # Step 3: Display editable dataframe with a checkbox for 'select' column
        edited_results = st.data_editor(results, column_config={"select": st.column_config.CheckboxColumn()})
        
        st.button("Get Data")

        # st.dataframe(results)
    else:
        st.warning("No results found.")
st.divider()

#########################



if st.session_state.connection:
    
    # Streamlit App

    
    
    query = '''
            SELECT 
                u.user_id AS user_id,
                u.name AS user_name,
                u.location AS user_location,
                u.institution AS user_institution,
                p.project_id AS project_id,
                p.project_name AS project_name,
                p.start_date AS project_start_date,
                p.source AS project_source,
                db.batch_id AS batch_id,
                db.upload_date AS batch_upload_date,
                db.analyte_name AS batch_analyte_name,
                db.buffer_solution AS batch_buffer_solution,
                db.instrument_details AS batch_instrument_details,
                db.wavelength AS batch_wavelength,
                db.power AS batch_power,
                db.concentration AS batch_concentration,
                db.concentration_units AS batch_concentration_units,
                db.accumulation_time AS batch_accumulation_time,
                db.experimental_procedure AS batch_experimental_procedure,
                db.substrate_type AS batch_substrate_type,
                db.substrate_material AS batch_substrate_material,
                db.preparation_conditions AS batch_preparation_conditions,
                db.data_type AS batch_data_type,
                db.notes AS batch_notes
            FROM
                "user" u
            JOIN
                project_user pu ON u.user_id = pu.user_id
            JOIN
                "project" p ON p.project_id = pu.project_id
            JOIN
                project_batch pb ON p.project_id = pb.project_id
            JOIN
                "databatch" db ON db.batch_id = pb.batch_id;
            '''
    # Execute the query
    cur.execute(query)
    # Fetch all results
    rows = cur.fetchall()
    # Iterate through the rows and print them
    columns = [
        'user_id', 'user_name', 'user_location', 'user_institution', 
        'project_id', 'project_name', 'project_start_date', 'project_source',
        'batch_id', 'batch_upload_date', 'batch_analyte_name', 
        'batch_buffer_solution', 'batch_instrument_details', 'batch_wavelength',
        'batch_power', 'batch_concentration', 'batch_concentration_units',
        'batch_accumulation_time', 'batch_experimental_procedure', 
        'batch_substrate_type', 'batch_substrate_material', 'batch_preparation_conditions', 
        'batch_data_type', 'batch_notes'
    ]

    # Create a DataFrame
    st.session_state.meta_df = pd.DataFrame(rows, columns=columns)
    st.write(st.session_state.meta_df)
    
    query = """
    SELECT * FROM "user";
    """
    cur.execute(query)
    rows = cur.fetchall()

    columns = [desc[0] for desc in cur.description]  # Get column names from the cursor
    st.session_state.user_df = pd.DataFrame(rows, columns=columns)
    
    st.write(st.session_state.user_df)
    st.session_state.result_vector = st.session_state.result_vector = st.session_state.user_df.apply(lambda row: f"{row['name']}", axis=1).tolist()
    st.session_state.result_vector.insert(0, "Create new user")
    
    # column_names = [desc[0] for desc in cur.description]
    # user_table = pd.DataFrame(rows, columns=column_names)
    # user_table['user_info'] = user_table.apply(lambda row: f"{row['name']} - {row['institution']}", axis=1)

    # Drop the original 'name' and 'institution' columns
    # user_table = user_table[['user_info', 'user_id']]

    # # Set the combined information into st.session_state
    # st.session_state.user_table = user_table
    # st.write(st.session_state.user_table)
    
    query = 'SELECT MAX("user_id") FROM "user"'
    cur.execute(query)
    st.session_state.largest_user_id = cur.fetchone()[0]
    
    query = """
            SELECT * FROM "project";
            """
    cur.execute(query)
    rows = cur.fetchall()

    columns = [desc[0] for desc in cur.description]  # Get column names from the cursor
    st.session_state.project_df = pd.DataFrame(rows, columns=columns)
    st.write(st.session_state.project_df)
    
    st.write("### User")
try: 
    

    user_selection = st.selectbox(
        "Select user of inserting the data",
        st.session_state.result_vector)

    if user_selection == "Create new user":
        with st.form("New_DB_user_creation"):
            st.write("New DB user creation")
            st.session_state.full_name = st.text_input("Full name (First name then last name)")
            st.session_state.location = st.selectbox("Your country", options = [country.name for country in pycountry.countries])
            st.session_state.institution = st.text_input("Your Institution")
            st.session_state.contact_info = st.text_input("Email address")
            # Every form must have a submit button.
            new_DB_user_submitted = st.form_submit_button("submit")
        if new_DB_user_submitted:
            
            conn = psycopg2.connect(
                dbname="SpectraGuruDB",
                user=st.session_state.user,
                password=st.session_state.passkey,
                host="localhost",  # Use "localhost" for local database
                port="5432"  # Default PostgreSQL port
            )
            cur = conn.cursor()
            
            st.session_state.new_user_id = st.session_state.largest_user_id + int(1)
            
            insert_query = """
                INSERT INTO "user" ("user_id", "name", "location", "institution", "contact_info")
                VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(insert_query, (st.session_state.new_user_id, 
                                    st.session_state.full_name, 
                                    st.session_state.location, 
                                    st.session_state.institution, 
                                    st.session_state.contact_info))

            # Commit the changes to the database
            conn.commit()
            st.session_state.selected_user_id = int(st.session_state.new_user_id)
    else:
        st.success(f"User selected as '{user_selection}'")
        
        st.write("### Project")
        
        user_row = st.session_state.user_df[st.session_state.user_df['name'] == user_selection]

        st.session_state.selected_user_id = user_row.iloc[0]['user_id']
        
        # st.write(st.session_state.selected_user_id)
        
        conn = psycopg2.connect(
                dbname="SpectraGuruDB",
                user=st.session_state.user,
                password=st.session_state.passkey,
                host="localhost",  # Use "localhost" for local database
                port="5432"  # Default PostgreSQL port
            )
        cur = conn.cursor()
        
        query = """
        SELECT pu.user_id, p.project_id, p.project_name
        FROM "project" p
        JOIN "project_user" pu ON pu.project_id = p.project_id
        WHERE pu.user_id = %s
        """
        cur.execute(query, (int(st.session_state.selected_user_id),))
        project_rows = cur.fetchall()

        # Convert the result to a DataFrame
        columns = [desc[0] for desc in cur.description]  # Get column names from the cursor
        st.session_state.project_df = pd.DataFrame(project_rows, columns=columns)
        
        # st.write(st.session_state.project_df)
        
        st.session_state.select_project_vector = st.session_state.project_df['project_name']
        st.session_state.select_project_vector = pd.concat([st.session_state.select_project_vector, pd.Series(["Create new project"])])
        project_selection = st.selectbox(
        "Select a project you wish to work on or create a new project",
        st.session_state.select_project_vector)
        
        if project_selection == "Create new project":
            with st.form("New_DB_project_creation"):
                st.write("New DB project creation")
                st.session_state.project_name = st.text_input("Project name")
                st.session_state.start_date = st.date_input("Project starting date",value="today")
                st.session_state.source = st.text_input("Project source/literature")
                # Every form must have a submit button.
                new_DB_project_submitted = st.form_submit_button("submit")
            if new_DB_project_submitted:
                
                conn = psycopg2.connect(
                    dbname="SpectraGuruDB",
                    user=st.session_state.user,
                    password=st.session_state.passkey,
                    host="localhost",  # Use "localhost" for local database
                    port="5432"  # Default PostgreSQL port
                )
                cur = conn.cursor()
                
                query = 'SELECT MAX("project_id") FROM "project"'
                cur.execute(query)
                st.session_state.largest_project_id = cur.fetchone()[0]
                st.session_state.new_project_id = int(st.session_state.largest_project_id) + int(1)
                
                
                insert_query = """
                    INSERT INTO "project" ("project_id", "start_date", "source", "project_name")
                    VALUES (%s, %s, %s, %s);
                    
                    INSERT INTO "project_user" ("project_id", "user_id")
                    VALUES (%s, %s);
                """
                cur.execute(insert_query, (int(st.session_state.new_project_id), 
                                        st.session_state.start_date, 
                                        st.session_state.source, 
                                        st.session_state.project_name,
                                        int(st.session_state.new_project_id), 
                                        int(st.session_state.selected_user_id)))

                # Commit the changes to the database
                conn.commit()
                
                # insert_query = """
                #     INSERT INTO "project_user" ("project_id", "user_id")
                #     VALUES (%s, %s)
                # """
                # cur.execute(insert_query, (st.session_state.new_project_id, 
                #                         st.session_state.selected_user_id))

                # # Commit the changes to the database
                # conn.commit()
                
                st.success("Project creation successful!")
        else:
            st.success(f"User selected as '{project_selection}'")
            project_row = st.session_state.project_df[st.session_state.project_df['project_name'] == project_selection]
                # st.write(st.session_state.project_df)
            st.session_state.selected_project_id = project_row.iloc[0]['project_id']
            
            # query = 'SELECT MAX("project_id") FROM "project"'
            # cur.execute(query)
            # st.session_state.largest_project_id = cur.fetchone()[0]
            # st.session_state.new_project_id = int(st.session_state.largest_project_id) + int(1)
            
            # st.session_state.selected_project_id = int(st.session_state.largest_project_id)
            
            st.write(st.session_state.selected_project_id)
            
            st.write("### Batch")
            
            st.checkbox("Is this a standard spectrum?",key="standard_spectrum", value=True)
            
            with st.form("New_DB_batch_creation"):
                st.write("New DB batch creation")
                st.session_state.upload_date = st.date_input("Batch upload date",value="today")
                st.session_state.analyte_name = st.text_input("Analyte name")
                st.session_state.buffer_solution = st.text_input("Buffer solution")
                st.session_state.spectrum_input = st.selectbox("Spectrum Input", options=["SERS","Raman","IR"])
                st.session_state.instrument_details = st.text_input("Instrument details")
                st.session_state.wavelength = st.number_input("Wavelength", min_value=0.0, format="%.2f")
                st.session_state.power = st.number_input("Power (mW)", min_value=0.0, format="%.2f")
                st.session_state.concentration = st.number_input("Concentration", min_value=0.0, format="%.2f")
                st.session_state.concentration_units = st.text_input("Concentration units")
                st.session_state.accumulation_time = st.number_input("Accumulation time (sec)", min_value=0.0, format="%.2f")
                st.session_state.experimental_procedure = st.text_area("Experimental procedure")
                st.session_state.substrate_type = st.text_input("Substrate type")
                st.session_state.substrate_material = st.text_input("Substrate material")
                st.session_state.preparation_conditions = st.text_area("Preparation conditions")
                if st.session_state.standard_spectrum:
                    st.session_state.data_type = st.selectbox("Data type", options=["Standard"])
                else:
                    st.session_state.data_type = st.selectbox("Data type", options=["Raw","Processed"])
                st.session_state.notes = st.text_area("Notes")
                # Every form must have a submit button.
                new_DB_batch_submitted = st.form_submit_button("submit")
            if new_DB_batch_submitted:
                
                conn = psycopg2.connect(
                    dbname="SpectraGuruDB",
                    user=st.session_state.user,
                    password=st.session_state.passkey,
                    host="localhost",  # Use "localhost" for local database
                    port="5432"  # Default PostgreSQL port
                )
                cur = conn.cursor()
                
                if st.session_state.standard_spectrum:
                    column_name = "batch_standard_id"
                    table_name = "databatch_standard"
                    ass_name = "project_batch_standard"
                else:
                    column_name = "batch_id"
                    table_name = "databatch"
                    ass_name = "project_batch"

                # Construct the query with sanitized inputs (if you're sure they are safe)
                query = f'SELECT MAX({column_name}) FROM {table_name}'
                cur.execute(query)
                st.session_state.largest_batch_id = cur.fetchone()[0]
                st.session_state.new_batch_id = st.session_state.largest_batch_id + int(1)
                
                query = """
                SELECT * FROM "project";
                """
                cur.execute(query)
                rows = cur.fetchall()

                # project_row = st.session_state.project_df[st.session_state.project_df['project_name'] == project_selection]
                # # st.write(st.session_state.project_df)
                # st.session_state.selected_project_id = project_row.iloc[0]['project_id']
                # st.session_state.selected_project_id = int(st.session_state.largest_project_id) + int(1)
                
                
                upload_date = st.session_state.upload_date
                if isinstance(upload_date, datetime.date):
                    upload_date = upload_date.strftime('%Y-%m-%d')
                
                insert_query = f"""
                INSERT INTO "{table_name}" ({column_name},
                    upload_date, analyte_name, buffer_solution, spectrum_input, 
                    instrument_details, wavelength, power, concentration, 
                    concentration_units, accumulation_time, experimental_procedure, 
                    substrate_type, substrate_material, preparation_conditions, 
                    data_type, notes,project_id, user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                
                INSERT INTO "{ass_name}" ("project_id", "{column_name}")
                VALUES (%s, %s);
                """
                values = (
                    int(st.session_state.new_batch_id),
                    upload_date,  # Properly formatted date as 'YYYY-MM-DD'
                    st.session_state.analyte_name,
                    st.session_state.buffer_solution,
                    st.session_state.spectrum_input,
                    st.session_state.instrument_details,
                    st.session_state.wavelength,
                    st.session_state.power,
                    st.session_state.concentration,
                    st.session_state.concentration_units,
                    st.session_state.accumulation_time,
                    st.session_state.experimental_procedure,
                    st.session_state.substrate_type,
                    st.session_state.substrate_material,
                    st.session_state.preparation_conditions,
                    st.session_state.data_type,
                    st.session_state.notes,
                    int(st.session_state.selected_project_id), 
                    int(st.session_state.selected_user_id),
                    int(st.session_state.selected_project_id), 
                    int(st.session_state.new_batch_id)
                )

                # Execute the query with the parameterized values
                cur.execute(insert_query, values)
                # Commit the changes to the database
                conn.commit()
                st.success("Batch creation successful!")
                
            st.write("### Data Uploader")
            
            with st.form("New_data_insert"):
                st.write("Upload data here")
                
                st.session_state.uploaded_files = st.file_uploader(
                    "Choose a CSV file", accept_multiple_files=False, type=['csv']
                )
                upload_mode = st.radio(
                    "How should this upload be handled?",
                    options=[
                        "Generate SQL dump only (do not insert into DB)",
                        "Insert into DB and also generate SQL dump"
                    ],
                    index=0
                )
                default_batch_standard_id = int(st.session_state.get("new_batch_id", 1))
                target_batch_standard_id = st.number_input(
                    "Target batch_standard_id for dump/export",
                    min_value=1,
                    value=default_batch_standard_id,
                    step=1
                )
                new_data_insert_submitted = st.form_submit_button("submit")
            if new_data_insert_submitted:
                if st.session_state.uploaded_files is None:
                    st.error("Please upload a CSV file before submitting.")
                else:
                    try:
                        df = pd.read_csv(st.session_state.uploaded_files, header=None)
                        if df.shape[1] != 2:
                            st.error("The file must have exactly two columns.")
                        else:
                            # Force numeric conversion to guarantee schema-safe inserts.
                            df[0] = pd.to_numeric(df[0], errors="raise")
                            df[1] = pd.to_numeric(df[1], errors="raise")

                            st.success("File is valid and all checks passed!")
                            st.write(df)

                            dump_id = f"dump_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                            spectrum_standard_name = st.session_state.uploaded_files.name

                            if upload_mode == "Insert into DB and also generate SQL dump":
                                conn = psycopg2.connect(
                                    dbname="SpectraGuruDB",
                                    user=st.session_state.user,
                                    password=st.session_state.passkey,
                                    host="localhost",
                                    port="5432"
                                )
                                cur = conn.cursor()

                                cur.execute('SELECT COALESCE(MAX("spectrum_standard_id"), 0) FROM "spectrum_standard"')
                                spectrum_standard_id = int(cur.fetchone()[0]) + 1

                                cur.execute('SELECT COALESCE(MAX("spectrum_data_standard_id"), 0) FROM "spectrum_data_standard"')
                                spectrum_data_standard_id = int(cur.fetchone()[0]) + 1

                                cur.execute(
                                    "INSERT INTO spectrum_standard (spectrum_standard_id, spectrum_name, batch_standard_id) VALUES (%s, %s, %s);",
                                    (spectrum_standard_id, spectrum_standard_name, int(target_batch_standard_id))
                                )
                                for _, row in df.iterrows():
                                    cur.execute(
                                        "INSERT INTO spectrum_data_standard (spectrum_data_standard_id, spectrum_standard_id, wavenumber, intensity) VALUES (%s, %s, %s, %s);",
                                        (int(spectrum_data_standard_id), int(spectrum_standard_id), float(row[0]), float(row[1]))
                                    )
                                    spectrum_data_standard_id += 1

                                conn.commit()
                                st.success("Data inserted into DB successfully.")
                            else:
                                spectrum_standard_id = int(datetime.datetime.utcnow().timestamp())
                                spectrum_data_standard_id = spectrum_standard_id + 1
                                st.info("DB insert skipped (dump-only mode).")

                            sql_dump_text = build_spectrum_dump_sql(
                                dump_id=dump_id,
                                batch_standard_id=int(target_batch_standard_id),
                                spectrum_standard_id=int(spectrum_standard_id),
                                spectrum_data_standard_start_id=int(spectrum_data_standard_id),
                                spectrum_name=spectrum_standard_name,
                                spectrum_df=df
                            )
                            dump_path, manifest_path = save_sql_dump_and_stage(
                                sql_text=sql_dump_text,
                                dump_id=dump_id,
                                source_filename=spectrum_standard_name
                            )

                            st.success(f"SQL dump generated with id: {dump_id}")
                            st.write(f"Saved dump file: {dump_path}")
                            st.write(f"Cloud staging manifest updated: {manifest_path}")
                            st.download_button(
                                "Download SQL dump",
                                data=sql_dump_text,
                                file_name=os.path.basename(dump_path),
                                mime="application/sql"
                            )
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

except Exception as e: 
    pass
    # st.warning(e)
