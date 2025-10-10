from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hashlib, psycopg2, json

app = FastAPI()

class DatasetProvenance(BaseModel):
    dataset_name: str
    version: str
    source: str
    owner: str
    license_type: str
    license_document_uri: str
    collection_basis: str
    consent_id: str | None = None
    pii_fields: bool
    anonymization_applied: bool
    purpose: str
    jurisdiction: str
    created_by: str

def compute_hash(record, prev_hash):
    data_str = json.dumps(record.dict(), sort_keys=True)
    return hashlib.sha256((data_str + (prev_hash or "")).encode()).hexdigest()

@app.post("/provenance/register")
def register_dataset(ds: DatasetProvenance):
    conn = psycopg2.connect("dbname=rbia")
    cur = conn.cursor()
    cur.execute("SELECT hash_curr FROM rbia_data_provenance ORDER BY id DESC LIMIT 1")
    prev = cur.fetchone()
    prev_hash = prev[0] if prev else None
    curr_hash = compute_hash(ds, prev_hash)
    cur.execute("""
        INSERT INTO rbia_data_provenance (
            dataset_id, dataset_name, version, source, owner,
            license_type, license_document_uri, collection_basis, consent_id,
            pii_fields, anonymization_applied, purpose, jurisdiction,
            hash_prev, hash_curr, created_by
        ) VALUES (
            md5(%s || %s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING dataset_id
    """, (ds.dataset_name, ds.version, ds.dataset_name, ds.version, ds.source,
          ds.owner, ds.license_type, ds.license_document_uri, ds.collection_basis,
          ds.consent_id, ds.pii_fields, ds.anonymization_applied, ds.purpose,
          ds.jurisdiction, prev_hash, curr_hash, ds.created_by))
    conn.commit()
    return {"dataset_id": cur.fetchone()[0], "hash": curr_hash}
