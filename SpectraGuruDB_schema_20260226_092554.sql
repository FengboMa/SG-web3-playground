--
-- PostgreSQL database dump
--

-- Dumped from database version 12.20
-- Dumped by pg_dump version 12.20

-- Started on 2026-02-26 09:25:54

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 221 (class 1255 OID 35982)
-- Name: fn_databatch_spectrum_count(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION "public"."fn_databatch_spectrum_count"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NEW.batch_id IS NOT NULL THEN
      UPDATE databatch
      SET spectrum_count = spectrum_count + 1
      WHERE batch_id = NEW.batch_id;
    END IF;
    RETURN NEW;

  ELSIF TG_OP = 'DELETE' THEN
    IF OLD.batch_id IS NOT NULL THEN
      UPDATE databatch
      SET spectrum_count = spectrum_count - 1
      WHERE batch_id = OLD.batch_id;
    END IF;
    RETURN OLD;

  ELSIF TG_OP = 'UPDATE' THEN
    -- only adjust if the batch_id changed (including NULL <-> non-NULL)
    IF OLD.batch_id IS DISTINCT FROM NEW.batch_id THEN
      IF OLD.batch_id IS NOT NULL THEN
        UPDATE databatch
        SET spectrum_count = spectrum_count - 1
        WHERE batch_id = OLD.batch_id;
      END IF;
      IF NEW.batch_id IS NOT NULL THEN
        UPDATE databatch
        SET spectrum_count = spectrum_count + 1
        WHERE batch_id = NEW.batch_id;
      END IF;
    END IF;
    RETURN NEW;
  END IF;
END;
$$;


--
-- TOC entry 222 (class 1255 OID 35989)
-- Name: fn_databatch_standard_spectrum_count(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION "public"."fn_databatch_standard_spectrum_count"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NEW.batch_standard_id IS NOT NULL THEN
      UPDATE databatch_standard
      SET spectrum_count = spectrum_count + 1
      WHERE batch_standard_id = NEW.batch_standard_id;
    END IF;
    RETURN NEW;

  ELSIF TG_OP = 'DELETE' THEN
    IF OLD.batch_standard_id IS NOT NULL THEN
      UPDATE databatch_standard
      SET spectrum_count = spectrum_count - 1
      WHERE batch_standard_id = OLD.batch_standard_id;
    END IF;
    RETURN OLD;

  ELSIF TG_OP = 'UPDATE' THEN
    IF OLD.batch_standard_id IS DISTINCT FROM NEW.batch_standard_id THEN
      IF OLD.batch_standard_id IS NOT NULL THEN
        UPDATE databatch_standard
        SET spectrum_count = spectrum_count - 1
        WHERE batch_standard_id = OLD.batch_standard_id;
      END IF;
      IF NEW.batch_standard_id IS NOT NULL THEN
        UPDATE databatch_standard
        SET spectrum_count = spectrum_count + 1
        WHERE batch_standard_id = NEW.batch_standard_id;
      END IF;
    END IF;
    RETURN NEW;
  END IF;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = "heap";

--
-- TOC entry 207 (class 1259 OID 19400)
-- Name: databatch; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."databatch" (
    "batch_id" integer NOT NULL,
    "upload_date" "date",
    "analyte_name" character varying(255),
    "buffer_solution" character varying(255),
    "spectrum_input" character varying(255),
    "instrument_details" character varying(255),
    "wavelength" double precision,
    "power" double precision,
    "concentration" double precision,
    "concentration_units" character varying(50),
    "accumulation_time" double precision,
    "experimental_procedure" "text",
    "substrate_type" character varying(255),
    "substrate_material" character varying(255),
    "preparation_conditions" "text",
    "data_type" character varying(255),
    "notes" "text",
    "project_id" integer,
    "user_id" integer,
    "spectrum_count" integer DEFAULT 0 NOT NULL,
    CONSTRAINT "databatch_spectrum_count_nonneg" CHECK (("spectrum_count" >= 0))
);


--
-- TOC entry 206 (class 1259 OID 19398)
-- Name: databatch_batch_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."databatch_batch_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2935 (class 0 OID 0)
-- Dependencies: 206
-- Name: databatch_batch_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."databatch_batch_id_seq" OWNED BY "public"."databatch"."batch_id";


--
-- TOC entry 215 (class 1259 OID 27691)
-- Name: databatch_standard; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."databatch_standard" (
    "batch_standard_id" integer NOT NULL,
    "upload_date" "date",
    "analyte_name" character varying(255),
    "buffer_solution" character varying(255),
    "spectrum_input" character varying(255),
    "instrument_details" character varying(255),
    "wavelength" double precision,
    "power" double precision,
    "concentration" double precision,
    "concentration_units" character varying(50),
    "accumulation_time" double precision,
    "experimental_procedure" "text",
    "substrate_type" character varying(255),
    "substrate_material" character varying(255),
    "preparation_conditions" "text",
    "data_type" character varying(255),
    "notes" "text",
    "project_id" integer,
    "user_id" integer,
    "spectrum_count" integer DEFAULT 0 NOT NULL,
    CONSTRAINT "databatch_standard_spectrum_count_nonneg" CHECK (("spectrum_count" >= 0))
);


--
-- TOC entry 214 (class 1259 OID 27689)
-- Name: databatch_standard_batch_standard_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."databatch_standard_batch_standard_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2936 (class 0 OID 0)
-- Dependencies: 214
-- Name: databatch_standard_batch_standard_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."databatch_standard_batch_standard_id_seq" OWNED BY "public"."databatch_standard"."batch_standard_id";


--
-- TOC entry 205 (class 1259 OID 19392)
-- Name: project; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."project" (
    "project_id" integer NOT NULL,
    "start_date" "date",
    "source" character varying(255),
    "project_name" character varying(255)
);


--
-- TOC entry 209 (class 1259 OID 19447)
-- Name: project_batch; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."project_batch" (
    "project_id" integer NOT NULL,
    "batch_id" integer NOT NULL
);


--
-- TOC entry 220 (class 1259 OID 27736)
-- Name: project_batch_standard; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."project_batch_standard" (
    "project_id" integer NOT NULL,
    "batch_standard_id" integer NOT NULL
);


--
-- TOC entry 204 (class 1259 OID 19390)
-- Name: project_project_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."project_project_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2937 (class 0 OID 0)
-- Dependencies: 204
-- Name: project_project_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."project_project_id_seq" OWNED BY "public"."project"."project_id";


--
-- TOC entry 208 (class 1259 OID 19432)
-- Name: project_user; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."project_user" (
    "project_id" integer NOT NULL,
    "user_id" integer NOT NULL
);


--
-- TOC entry 211 (class 1259 OID 19464)
-- Name: spectrum; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."spectrum" (
    "spectrum_id" integer NOT NULL,
    "spectrum_name" character varying(255),
    "batch_id" integer
);


--
-- TOC entry 213 (class 1259 OID 19477)
-- Name: spectrum_data; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."spectrum_data" (
    "spectrum_data_id" integer NOT NULL,
    "spectrum_id" integer,
    "wavenumber" double precision,
    "intensity" double precision
);


--
-- TOC entry 212 (class 1259 OID 19475)
-- Name: spectrum_data_spectrum_data_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."spectrum_data_spectrum_data_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2938 (class 0 OID 0)
-- Dependencies: 212
-- Name: spectrum_data_spectrum_data_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."spectrum_data_spectrum_data_id_seq" OWNED BY "public"."spectrum_data"."spectrum_data_id";


--
-- TOC entry 219 (class 1259 OID 27725)
-- Name: spectrum_data_standard; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."spectrum_data_standard" (
    "spectrum_data_standard_id" integer NOT NULL,
    "spectrum_standard_id" integer,
    "wavenumber" double precision,
    "intensity" double precision
);


--
-- TOC entry 218 (class 1259 OID 27723)
-- Name: spectrum_data_standard_spectrum_data_standard_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."spectrum_data_standard_spectrum_data_standard_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2939 (class 0 OID 0)
-- Dependencies: 218
-- Name: spectrum_data_standard_spectrum_data_standard_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."spectrum_data_standard_spectrum_data_standard_id_seq" OWNED BY "public"."spectrum_data_standard"."spectrum_data_standard_id";


--
-- TOC entry 210 (class 1259 OID 19462)
-- Name: spectrum_spectrum_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."spectrum_spectrum_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2940 (class 0 OID 0)
-- Dependencies: 210
-- Name: spectrum_spectrum_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."spectrum_spectrum_id_seq" OWNED BY "public"."spectrum"."spectrum_id";


--
-- TOC entry 217 (class 1259 OID 27712)
-- Name: spectrum_standard; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."spectrum_standard" (
    "spectrum_standard_id" integer NOT NULL,
    "spectrum_name" character varying(255),
    "batch_standard_id" integer
);


--
-- TOC entry 216 (class 1259 OID 27710)
-- Name: spectrum_standard_spectrum_standard_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."spectrum_standard_spectrum_standard_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2941 (class 0 OID 0)
-- Dependencies: 216
-- Name: spectrum_standard_spectrum_standard_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."spectrum_standard_spectrum_standard_id_seq" OWNED BY "public"."spectrum_standard"."spectrum_standard_id";


--
-- TOC entry 203 (class 1259 OID 19381)
-- Name: user; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE "public"."user" (
    "user_id" integer NOT NULL,
    "name" character varying(255),
    "location" character varying(255),
    "institution" character varying(255),
    "contact_info" character varying(255)
);


--
-- TOC entry 202 (class 1259 OID 19379)
-- Name: user_user_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE "public"."user_user_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 2942 (class 0 OID 0)
-- Dependencies: 202
-- Name: user_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE "public"."user_user_id_seq" OWNED BY "public"."user"."user_id";


--
-- TOC entry 2750 (class 2604 OID 19403)
-- Name: databatch batch_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch" ALTER COLUMN "batch_id" SET DEFAULT "nextval"('"public"."databatch_batch_id_seq"'::"regclass");


--
-- TOC entry 2755 (class 2604 OID 27694)
-- Name: databatch_standard batch_standard_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch_standard" ALTER COLUMN "batch_standard_id" SET DEFAULT "nextval"('"public"."databatch_standard_batch_standard_id_seq"'::"regclass");


--
-- TOC entry 2749 (class 2604 OID 19395)
-- Name: project project_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project" ALTER COLUMN "project_id" SET DEFAULT "nextval"('"public"."project_project_id_seq"'::"regclass");


--
-- TOC entry 2753 (class 2604 OID 19467)
-- Name: spectrum spectrum_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum" ALTER COLUMN "spectrum_id" SET DEFAULT "nextval"('"public"."spectrum_spectrum_id_seq"'::"regclass");


--
-- TOC entry 2754 (class 2604 OID 19480)
-- Name: spectrum_data spectrum_data_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_data" ALTER COLUMN "spectrum_data_id" SET DEFAULT "nextval"('"public"."spectrum_data_spectrum_data_id_seq"'::"regclass");


--
-- TOC entry 2759 (class 2604 OID 27728)
-- Name: spectrum_data_standard spectrum_data_standard_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_data_standard" ALTER COLUMN "spectrum_data_standard_id" SET DEFAULT "nextval"('"public"."spectrum_data_standard_spectrum_data_standard_id_seq"'::"regclass");


--
-- TOC entry 2758 (class 2604 OID 27715)
-- Name: spectrum_standard spectrum_standard_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_standard" ALTER COLUMN "spectrum_standard_id" SET DEFAULT "nextval"('"public"."spectrum_standard_spectrum_standard_id_seq"'::"regclass");


--
-- TOC entry 2748 (class 2604 OID 19384)
-- Name: user user_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."user" ALTER COLUMN "user_id" SET DEFAULT "nextval"('"public"."user_user_id_seq"'::"regclass");


--
-- TOC entry 2765 (class 2606 OID 19408)
-- Name: databatch databatch_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch"
    ADD CONSTRAINT "databatch_pkey" PRIMARY KEY ("batch_id");


--
-- TOC entry 2776 (class 2606 OID 27699)
-- Name: databatch_standard databatch_standard_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch_standard"
    ADD CONSTRAINT "databatch_standard_pkey" PRIMARY KEY ("batch_standard_id");


--
-- TOC entry 2769 (class 2606 OID 19451)
-- Name: project_batch project_batch_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_batch"
    ADD CONSTRAINT "project_batch_pkey" PRIMARY KEY ("project_id", "batch_id");


--
-- TOC entry 2783 (class 2606 OID 27740)
-- Name: project_batch_standard project_batch_standard_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_batch_standard"
    ADD CONSTRAINT "project_batch_standard_pkey" PRIMARY KEY ("project_id", "batch_standard_id");


--
-- TOC entry 2763 (class 2606 OID 19397)
-- Name: project project_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project"
    ADD CONSTRAINT "project_pkey" PRIMARY KEY ("project_id");


--
-- TOC entry 2767 (class 2606 OID 19436)
-- Name: project_user project_user_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_user"
    ADD CONSTRAINT "project_user_pkey" PRIMARY KEY ("project_id", "user_id");


--
-- TOC entry 2774 (class 2606 OID 19482)
-- Name: spectrum_data spectrum_data_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_data"
    ADD CONSTRAINT "spectrum_data_pkey" PRIMARY KEY ("spectrum_data_id");


--
-- TOC entry 2781 (class 2606 OID 27730)
-- Name: spectrum_data_standard spectrum_data_standard_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_data_standard"
    ADD CONSTRAINT "spectrum_data_standard_pkey" PRIMARY KEY ("spectrum_data_standard_id");


--
-- TOC entry 2772 (class 2606 OID 19469)
-- Name: spectrum spectrum_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum"
    ADD CONSTRAINT "spectrum_pkey" PRIMARY KEY ("spectrum_id");


--
-- TOC entry 2779 (class 2606 OID 27717)
-- Name: spectrum_standard spectrum_standard_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_standard"
    ADD CONSTRAINT "spectrum_standard_pkey" PRIMARY KEY ("spectrum_standard_id");


--
-- TOC entry 2761 (class 2606 OID 19389)
-- Name: user user_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."user"
    ADD CONSTRAINT "user_pkey" PRIMARY KEY ("user_id");


--
-- TOC entry 2770 (class 1259 OID 35973)
-- Name: idx_spectrum_batch_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_spectrum_batch_id" ON "public"."spectrum" USING "btree" ("batch_id");


--
-- TOC entry 2777 (class 1259 OID 35986)
-- Name: idx_spectrum_standard_batch_standard_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX "idx_spectrum_standard_batch_standard_id" ON "public"."spectrum_standard" USING "btree" ("batch_standard_id");


--
-- TOC entry 2798 (class 2620 OID 35984)
-- Name: spectrum trg_spectrum_count_del; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_spectrum_count_del" AFTER DELETE ON "public"."spectrum" FOR EACH ROW EXECUTE FUNCTION "public"."fn_databatch_spectrum_count"();


--
-- TOC entry 2799 (class 2620 OID 35983)
-- Name: spectrum trg_spectrum_count_ins; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_spectrum_count_ins" AFTER INSERT ON "public"."spectrum" FOR EACH ROW EXECUTE FUNCTION "public"."fn_databatch_spectrum_count"();


--
-- TOC entry 2800 (class 2620 OID 35985)
-- Name: spectrum trg_spectrum_count_upd; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_spectrum_count_upd" AFTER UPDATE OF "batch_id" ON "public"."spectrum" FOR EACH ROW EXECUTE FUNCTION "public"."fn_databatch_spectrum_count"();


--
-- TOC entry 2801 (class 2620 OID 35991)
-- Name: spectrum_standard trg_spectrum_standard_count_del; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_spectrum_standard_count_del" AFTER DELETE ON "public"."spectrum_standard" FOR EACH ROW EXECUTE FUNCTION "public"."fn_databatch_standard_spectrum_count"();


--
-- TOC entry 2802 (class 2620 OID 35990)
-- Name: spectrum_standard trg_spectrum_standard_count_ins; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_spectrum_standard_count_ins" AFTER INSERT ON "public"."spectrum_standard" FOR EACH ROW EXECUTE FUNCTION "public"."fn_databatch_standard_spectrum_count"();


--
-- TOC entry 2803 (class 2620 OID 35992)
-- Name: spectrum_standard trg_spectrum_standard_count_upd; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER "trg_spectrum_standard_count_upd" AFTER UPDATE OF "batch_standard_id" ON "public"."spectrum_standard" FOR EACH ROW EXECUTE FUNCTION "public"."fn_databatch_standard_spectrum_count"();


--
-- TOC entry 2784 (class 2606 OID 19409)
-- Name: databatch databatch_project_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch"
    ADD CONSTRAINT "databatch_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."project"("project_id");


--
-- TOC entry 2792 (class 2606 OID 27700)
-- Name: databatch_standard databatch_standard_project_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch_standard"
    ADD CONSTRAINT "databatch_standard_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."project"("project_id");


--
-- TOC entry 2793 (class 2606 OID 27705)
-- Name: databatch_standard databatch_standard_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch_standard"
    ADD CONSTRAINT "databatch_standard_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."user"("user_id");


--
-- TOC entry 2785 (class 2606 OID 19414)
-- Name: databatch databatch_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."databatch"
    ADD CONSTRAINT "databatch_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."user"("user_id");


--
-- TOC entry 2789 (class 2606 OID 19457)
-- Name: project_batch project_batch_batch_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_batch"
    ADD CONSTRAINT "project_batch_batch_id_fkey" FOREIGN KEY ("batch_id") REFERENCES "public"."databatch"("batch_id");


--
-- TOC entry 2788 (class 2606 OID 19452)
-- Name: project_batch project_batch_project_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_batch"
    ADD CONSTRAINT "project_batch_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."project"("project_id");


--
-- TOC entry 2797 (class 2606 OID 27746)
-- Name: project_batch_standard project_batch_standard_batch_standard_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_batch_standard"
    ADD CONSTRAINT "project_batch_standard_batch_standard_id_fkey" FOREIGN KEY ("batch_standard_id") REFERENCES "public"."databatch_standard"("batch_standard_id");


--
-- TOC entry 2796 (class 2606 OID 27741)
-- Name: project_batch_standard project_batch_standard_project_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_batch_standard"
    ADD CONSTRAINT "project_batch_standard_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."project"("project_id");


--
-- TOC entry 2786 (class 2606 OID 19437)
-- Name: project_user project_user_project_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_user"
    ADD CONSTRAINT "project_user_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."project"("project_id");


--
-- TOC entry 2787 (class 2606 OID 19442)
-- Name: project_user project_user_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."project_user"
    ADD CONSTRAINT "project_user_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."user"("user_id");


--
-- TOC entry 2790 (class 2606 OID 19470)
-- Name: spectrum spectrum_batch_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum"
    ADD CONSTRAINT "spectrum_batch_id_fkey" FOREIGN KEY ("batch_id") REFERENCES "public"."databatch"("batch_id");


--
-- TOC entry 2791 (class 2606 OID 19483)
-- Name: spectrum_data spectrum_data_spectrum_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_data"
    ADD CONSTRAINT "spectrum_data_spectrum_id_fkey" FOREIGN KEY ("spectrum_id") REFERENCES "public"."spectrum"("spectrum_id");


--
-- TOC entry 2795 (class 2606 OID 27731)
-- Name: spectrum_data_standard spectrum_data_standard_spectrum_standard_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_data_standard"
    ADD CONSTRAINT "spectrum_data_standard_spectrum_standard_id_fkey" FOREIGN KEY ("spectrum_standard_id") REFERENCES "public"."spectrum_standard"("spectrum_standard_id");


--
-- TOC entry 2794 (class 2606 OID 27718)
-- Name: spectrum_standard spectrum_standard_batch_standard_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY "public"."spectrum_standard"
    ADD CONSTRAINT "spectrum_standard_batch_standard_id_fkey" FOREIGN KEY ("batch_standard_id") REFERENCES "public"."databatch_standard"("batch_standard_id");


-- Completed on 2026-02-26 09:25:54

--
-- PostgreSQL database dump complete
--

