from flask import Flask, request, jsonify, send_from_directory
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import os
from parser import parse_resume
from scorer import ResumeScorer
from datetime import datetime
import uuid
import docx
import hashlib
import logging
from zlib import compress;
from io import BytesIO

# Configure logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")

# CORS configuration
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Serve index.html for the root route
@app.route("/")
def serve_index():
    return send_from_directory(app.template_folder, "index.html")

# Serve favicon.ico
@app.route("/favicon.ico")
def serve_favicon():
    return send_from_directory(os.path.join(app.root_path, "../images"), "favicon.ico")

# Serve other static files (e.g., style.css, script.js)
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client[os.getenv("DB_NAME")]
fs = GridFS(db)

# Check if the 'resumes' collection exists, create it if it doesn't
if "resumes" not in db.list_collection_names():
    db.create_collection("resumes")
resumes_collection = db["resumes"]

# Default job keywords (can be customized per request)
DEFAULT_KEYWORDS = [
    # General Keywords
    "python", "machine learning", "data analysis", "communication", "teamwork", "problem solving",
    "project management", "agile", "scrum", "devops", "cybersecurity",

    # Infrastructure
    "active directory", "group policy objects", "gpo", "windows server", "exchange online",
    "adfs", "active directory federation services", "distributed file system", "dfs",
    "hyper-v", "vmware esxi", "citrix", "sccm", "system center configuration manager",

    # Cloud & Virtualization
    "azure iaas", "storsimple", "twinstrata", "veeam", "vmware vsphere", "virtual desktop infrastructure", "vdi",
    "microsoft azure", "cloud array", "storage management", "enterprise backup",
    "aws ec2", "aws s3", "google cloud platform", "gcp", "kubernetes", "docker swarm",
    "terraform", "ansible", "cloudformation",

    # Networking & Security
    "tcp/ip", "dns", "dhcp", "vpn", "cisco asa", "firewall", "isa server", "proxy server",
    "network security", "wireless lan", "certificates", "mcafee epolicy orchestrator", "epo",
    "palo alto", "fortinet", "siem", "splunk", "wireshark", "penetration testing",
    "encryption", "oauth", "saml", "ids/ips", "intrusion detection",

    # Enterprise Tools
    "solarwinds", "system center configuration manager", "sccm", "symantec backup exec", "veritas netbackup",
    "data domain", "enterprise vault", "sharepoint", "iis", "team foundation server", "tfs",
    "jira", "confluence", "servicenow", "nagios", "zabbix", "prometheus", "grafana",

    # Scripting & Programming
    "powershell", "vbscript", "bash", "linux", "ubuntu", "windows 7/8/10", "windows server 2003/2008/2012",
    "javascript", "typescript", "node.js", "react", "angular", "java", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "graphql",

    # Certifications (Explicit)
    "comptia network+", "microsoft certified", "certified", "cissp", "ccna", "ccnp",
    "aws certified", "azure certified", "vmware certified professional", "vcp",
    "red hat certified", "rhce", "comptia security+", "ceh", "certified ethical hacker"
]

# Supported file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file, filename):
    try:
        extension = filename.rsplit('.', 1)[1].lower()
        logger.debug(f"Validating file: {filename}, extension: {extension}")
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024: 
            return False, "File size exceeds 10MB limit."
        if file_size == 0:
            return False, "File is empty."
            
        # Extension-specific validation
        if extension == 'pdf':
            header = file.read(4)
            file.seek(0)
            if header != b'%PDF':
                return False, "Invalid PDF file (missing PDF header)"
                
        elif extension == 'docx':
            try:
                doc = docx.Document(file)
                if not doc.paragraphs:
                    return False, "Empty DOCX file"
                file.seek(0)
            except:
                return False, "Invalid DOCX file"
                
        elif extension == 'txt':
            try:
                content = file.read().decode('utf-8', errors='ignore')
                if not content.strip():
                    return False, "Empty text file"
                file.seek(0)
            except:
                return False, "Invalid text file"
                
        return True, ""
        
    except Exception as e:
        logger.error(f"File validation failed: {str(e)}")
        return False, f"Invalid file: {str(e)}"

def generate_resume_group_id(filename, job_title):
    """Generate a unique resume group ID based on filename and job title."""
    combined = f"{filename}:{job_title}".encode('utf-8')
    return hashlib.sha256(combined).hexdigest()

@app.route("/api/upload", methods=["POST", "OPTIONS"])
def upload_resume():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    job_title = request.form.get("job_title", "")
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a PDF, DOCX, or TXT file."}), 400
    
    # Validate file content
    is_valid, error_message = validate_file(file, file.filename)
    if not is_valid:
        return jsonify({"error": error_message}), 400
    
    try:
        # Generate unique filename and resume group ID
        file_id = str(uuid.uuid4())
        resume_group_id = generate_resume_group_id(file.filename, job_title)
        extension = file.filename.rsplit('.', 1)[1].lower()
        
        # Store file in GridFS
        file.seek(0)
        gridfs_id = fs.put(file, filename=f"{file_id}.{extension}")
        
        # Parse resume from GridFS with proper filename attribute
        file_data = fs.get(gridfs_id)
        file_obj = BytesIO(file_data.read())
        file_obj.filename = f"{file_id}.{extension}"
        file_obj.seek(0)
                
        # Get the latest version number for this resume group
        latest_version = resumes_collection.find_one(
            {"resume_group_id": resume_group_id},
            sort=[("version_number", -1)]
        )
        next_version = 1 if not latest_version else latest_version["version_number"] + 1
        
        resume_data = parse_resume(file_obj)
        
        # Score resume
        keywords = DEFAULT_KEYWORDS
        if job_title:
            keywords.extend(job_title.lower().split())
            
        scorer = ResumeScorer(job_keywords=keywords, cohere_api_key=os.getenv("COHERE_API_KEY"))
        score, score_breakdown, tone_analysis = scorer.calculate_score(resume_data)
        feedback = scorer.generate_feedback(resume_data, score_breakdown)
        
        # Create indexes if they don't exist (optimization)
        if "resume_group_id_1_version_number_-1" not in resumes_collection.index_information():
            resumes_collection.create_index([
                ("resume_group_id", 1), 
                ("version_number", -1)
            ], name="resume_group_id_1_version_number_-1")
            
        if "file_id_1" not in resumes_collection.index_information():
            resumes_collection.create_index([("file_id", 1)], name="file_id_1")
        
        # Store in MongoDB
        resume_record = {
            "file_id": file_id,
            "resume_group_id": resume_group_id,
            "version_number": next_version,  # Use calculated version number
            "original_filename": file.filename,
            "upload_date": datetime.utcnow(),
            "score": score,
            "feedback": feedback,
            "tone_analysis": tone_analysis,
            "resume_data": compress(str(resume_data).encode()),
            "job_title": job_title,
            "gridfs_id": str(gridfs_id)
        }
        
        resumes_collection.insert_one(resume_record)
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "resume_group_id": resume_group_id,
            "version_number": next_version,  # Return actual version number
            "score": score,
            "feedback": feedback,
            "tone_analysis": tone_analysis
        }), 200
        
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid pdf" in error_msg or "scanned pdf" in error_msg:
            return jsonify({"error": "Invalid PDF file. Please upload a valid, text-based PDF."}), 400
        logger.exception("Resume upload failed")
        return jsonify({"error": "An error occurred while processing your file"}), 500
    

@app.route("/api/resume/<file_id>", methods=["GET"])
def get_resume(file_id):
    resume = resumes_collection.find_one({"file_id": file_id})
    if not resume:
        return jsonify({"error": "Resume not found"}), 404
    
    # Remove MongoDB _id field
    resume.pop("_id", None)
    return jsonify(resume), 200

@app.route("/api/history/<resume_group_id>", methods=["GET"])
def get_version_history(resume_group_id):
    try:
        page = int(request.args.get('page', 1))
        per_page = 10
        
        # Get total count for pagination info
        total_count = resumes_collection.count_documents({"resume_group_id": resume_group_id})
        
        # Get paginated results
        versions = resumes_collection.find(
            {"resume_group_id": resume_group_id}
        ).sort("version_number", 1).skip((page-1)*per_page).limit(per_page)
        
        version_list = []
        for version in versions:
            version_list.append({
                "file_id": version["file_id"],
                "version_number": version["version_number"],
                "upload_date": version["upload_date"].isoformat(),
                "score": version["score"],
                "feedback": version["feedback"],
                "tone_analysis": version["tone_analysis"],
                "job_title": version["job_title"]
            })
        
        if not version_list:
            return jsonify({"error": "No version history found for this resume group"}), 404
            
        return jsonify({
            "success": True,
            "versions": version_list,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": (total_count + per_page - 1) // per_page
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching version history: {str(e)}")
        return jsonify({"error": "An error occurred while fetching version history"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)