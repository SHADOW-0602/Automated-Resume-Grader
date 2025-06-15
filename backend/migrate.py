import os
import subprocess
from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def install_spacy_model():
    try:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
        logger.info("spaCy model 'en_core_web_lg' installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install spaCy model: {str(e)}")
        raise

def connect_to_mongodb():
    try:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI not set in environment variables")
        client = MongoClient(mongo_uri)
        db_name = os.getenv("DB_NAME")
        if not db_name:
            raise ValueError("DB_NAME not set in environment variables")
        db = client[db_name]
        logger.info(f"Connected to MongoDB database: {db_name}")
        return client, db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

def create_indexes(db):
    try:
        resumes_collection = db["resumes"]
        if "resumes" not in db.list_collection_names():
            db.create_collection("resumes")
            logger.info("Created 'resumes' collection")
        index_name = "resume_group_id_1_upload_date_-1"
        existing_indexes = resumes_collection.index_information()
        if index_name not in existing_indexes:
            resumes_collection.create_index(
                [("resume_group_id", 1), ("upload_date", DESCENDING)],
                name=index_name,
                background=True  # Background indexing for non-blocking
            )
            logger.info(f"Created index: {index_name}")
        else:
            logger.info(f"Index {index_name} already exists, skipping creation")
        file_id_index_name = "file_id_1"
        if file_id_index_name not in existing_indexes:
            resumes_collection.create_index(
                [("file_id", 1)],
                name=file_id_index_name,
                unique=True
            )
            logger.info(f"Created index: {file_id_index_name}")
        else:
            logger.info(f"Index {file_id_index_name} already exists, skipping creation")
    except Exception as e:
        logger.error(f"Failed to create indexes: {str(e)}")
        raise

def run_migration():
    try:
        install_spacy_model()
        client, db = connect_to_mongodb()
        create_indexes(db)
        logger.info("Migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise
    finally:
        client.close()
        logger.info("MongoDB connection closed")

if __name__ == "__main__":
    run_migration()