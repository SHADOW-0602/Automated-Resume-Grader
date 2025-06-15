import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import cohere
import logging
from dotenv import load_dotenv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import language_tool_python
from typing import Dict, List, Tuple
from textstat import flesch_kincaid_grade;

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

class ResumeScorer:
    def __init__(self, job_keywords=None, cohere_api_key=None, historical_data=None):
        self.job_keywords = {kw.lower() for kw in (job_keywords or [])}
        self.sid = SentimentIntensityAnalyzer()
        
        if cohere_api_key:
            self.co = cohere.Client(cohere_api_key)
            self.keyword_embeddings = {}  # Cache for keyword embeddings
        else:
            logger.warning("Cohere API key not provided; some features will be limited")
            self.co = None
        
        # Initialize language tool for grammar checking
        self.language_tool = language_tool_python.LanguageTool('en-US')
        
        # Define default weights
        self.default_weights = {
            "section_completeness": 15,
            "keyword_relevance": 25,
            "experience_relevance": 20,
            "education_match": 10,
            "skills_depth": 15,
            "formatting": 5,
            "clarity": 5,
            "contact_info": 3,
            "tone_positivity": 2
        }

        # Example transformations for feedback
        self.EXAMPLE_TRANSFORMATIONS = {
            "weak_verbs": [
                {
                    "before": "Was responsible for network maintenance.",
                    "after": "Spearheaded network maintenance, reducing downtime by 25%."
                },
                {
                    "before": "Helped with cloud migration.",
                    "after": "Orchestrated cloud migration, completing project 2 weeks ahead of schedule."
                },
                {
                    "before": "Did system upgrades.",
                    "after": "Engineered system upgrades, improving performance by 30%."
                }
            ],
            "negative_phrases": [
                {
                    "before": "Failed to meet project deadline.",
                    "after": "Identified process gaps and implemented improvements to meet future deadlines."
                },
                {
                    "before": "Faced challenges with system integration.",
                    "after": "Overcame system integration challenges by developing a custom API."
                }
            ],
            "passive_phrases": [
                {
                    "before": "Tasks were completed on time.",
                    "after": "Completed tasks on time, ensuring project milestones were met."
                },
                {
                    "before": "The system was managed by me.",
                    "after": "Managed the system, ensuring 99.9% uptime."
                }
            ]
        }
        
        # Initialize with default weights
        self.weights = self.default_weights.copy()
        
        # If historical data is available, train weights
        if historical_data:
            self._train_weights(historical_data)
        
        # Certification and skill configurations
        self.CERTIFICATION_TIERS = {
            "High": ["cissp", "ccnp", "aws certified solutions architect - professional"],
            "Medium": ["ccna", "comptia security+"],
            "Low": ["comptia a+", "microsoft office specialist"],
            "Outdated": ["microsoft frontpage", "novell netware", "mcse nt 4.0", "a+ 2003"]
        }
        
        self.SKILL_CLUSTERS = {
            "Cloud": ["azure", "aws", "gcp", "iaas", "storsimple"],
            "Security": ["firewall", "vpn", "encryption", "mcafee epolicy orchestrator"]
        }
        
        self.PRESTIGE_SCHOOLS = ["mit", "stanford", "harvard", "caltech"]
        
        # Ensure total weights sum to 100
        total_weight_sum = sum(self.weights.values())
        if total_weight_sum != 100:
            factor = 100 / total_weight_sum
            self.weights = {k: v * factor for k, v in self.weights.items()}

    def _train_weights(self, historical_data):
        """Train weights using logistic regression on historical data."""
        try:
            # Prepare features and labels
            X = []
            y = []
            
            for data in historical_data:
                features = [
                    data.get("section_completeness", 0),
                    data.get("keyword_relevance", 0),
                    data.get("experience_relevance", 0),
                    data.get("education_match", 0),
                    data.get("skills_depth", 0),
                    data.get("formatting", 0),
                    data.get("clarity", 0),
                    data.get("contact_info", 0),
                    data.get("tone_positivity", 0)
                ]
                X.append(features)
                y.append(1 if data.get("success", False) else 0)
            
            if len(X) < 10:
                logger.warning("Insufficient historical data for training, using default weights")
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train logistic regression model
            model = LogisticRegression(penalty='l1', solver='liblinear')
            model.fit(X_scaled, y)
            
            # Extract and normalize weights
            learned_weights = np.abs(model.coef_[0])
            learned_weights = (learned_weights / learned_weights.sum()) * 100
            
            # Update weights dictionary
            self.weights = {
                "section_completeness": learned_weights[0],
                "keyword_relevance": learned_weights[1],
                "experience_relevance": learned_weights[2],
                "education_match": learned_weights[3],
                "skills_depth": learned_weights[4],
                "formatting": learned_weights[5],
                "clarity": learned_weights[6],
                "contact_info": learned_weights[7],
                "tone_positivity": learned_weights[8]
            }
            
            logger.info(f"Updated weights based on historical data: {self.weights}")
            
        except Exception as e:
            logger.error(f"Failed to train weights: {str(e)}")
            self.weights = self.default_weights.copy()        

    def calculate_score(self, resume_data):
        score_breakdown = {
            "section_completeness": self.score_section_completeness(resume_data),
            "keyword_relevance": self.score_keyword_relevance(resume_data),
            "experience_relevance": self.score_experience_relevance(resume_data),
            "education_match": self.score_education_match(resume_data),
            "skills_depth": self.score_skills_depth(resume_data),
            "formatting": self.score_formatting(resume_data),
            "clarity": self.score_clarity(resume_data),
            "contact_info": self.score_contact_info(resume_data),
            "tone_positivity": self.score_tone_positivity(resume_data)
        }
        
        # Apply weights to get final weighted scores for each category
        weighted_score_breakdown = {
            k: (v / 100) * self.weights[k] for k, v in score_breakdown.items()
        }

        total_score = sum(weighted_score_breakdown.values())
        
        # Generate tone analysis feedback
        tone_analysis = self.generate_tone_feedback(resume_data, score_breakdown["tone_positivity"])
        
        return min(100, round(total_score, 2)), weighted_score_breakdown, tone_analysis

    def score_certification_tier(self, certs):
        """Score certifications based on their tier."""
        score = 0
        for cert in certs:
            cert_lower = cert.lower()
            for tier, tier_certs in self.CERTIFICATION_TIERS.items():
                if any(tier_cert in cert_lower for tier_cert in tier_certs):
                    if tier == "High":
                        score += 10
                    elif tier == "Medium":
                        score += 5
                    elif tier == "Low":
                        score += 2
                    elif tier == "Outdated":
                        score += 1
        return min(20, score)

    def score_skills_clusters(self, skills):
        """Score skills based on clusters."""
        cluster_bonus = 0
        for cluster, terms in self.SKILL_CLUSTERS.items():
            if sum(term.lower() in skill.lower() for skill in skills for term in terms) >= 2:
                cluster_bonus += 10  # Bonus per complete cluster
        return min(30, cluster_bonus)

    def score_education_prestige(self, education_section):
        """Score education based on prestigious schools."""
        education_text = " ".join(education_section).lower()
        return 10 if any(school in education_text for school in self.PRESTIGE_SCHOOLS) else 0

    def adjust_weights_for_job(self, job_title):
        """Adjust scoring weights based on job title."""
        job_title_lower = job_title.lower()
        if "cloud" in job_title_lower:
            self.weights["skills_depth"] = 25  # Up from 15
            self.weights["keyword_relevance"] = 30  # Up from 25
            # Re-normalize weights
            total_weight_sum = sum(self.weights.values())
            factor = 100 / total_weight_sum
            self.weights = {k: v * factor for k, v in self.weights.items()}

    def expand_keywords_with_cohere(self, text_chunk: str) -> List[str]:
        """Expand keywords using Cohere embeddings with cosine similarity."""
        if not self.co or not self.job_keywords:
            return list(self.job_keywords)
        
        try:
            # Get embeddings for job keywords if not cached
            if not self.keyword_embeddings:
                keyword_list = list(self.job_keywords)
                response = self.co.embed(
                    texts=keyword_list,
                    model="embed-english-v3.0",
                    input_type="classification"
                )
                for kw, emb in zip(keyword_list, response.embeddings):
                    self.keyword_embeddings[kw] = np.array(emb)
            
            # Get embedding for the text chunk
            chunk_response = self.co.embed(
                texts=[text_chunk],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            chunk_embedding = np.array(chunk_response.embeddings[0])
            
            # Calculate cosine similarities
            expanded_keywords = set(self.job_keywords)
            similarity_threshold = 0.5
            
            for kw, kw_emb in self.keyword_embeddings.items():
                similarity = np.dot(chunk_embedding, kw_emb) / (
                    np.linalg.norm(chunk_embedding) * np.linalg.norm(kw_emb)
                )
                if similarity > similarity_threshold:
                    expanded_keywords.add(kw)
                    logger.debug(f"Added similar keyword: {kw} (similarity: {similarity:.2f})")
            
            return list(expanded_keywords)
            
        except Exception as e:
            logger.error(f"Cohere embedding failed: {str(e)}")
            return list(self.job_keywords)

    def score_section_completeness(self, resume_data):
        sections = resume_data.get("sections", {})
        score = 0
        
        # Base points for essential sections
        if sections.get("summary") and len(sections["summary"].strip()) > 50: score += 15
        if sections.get("experience") and len(sections["experience"]) >= 1: score += 20
        if sections.get("education") and len(sections["education"]) >= 1: score += 15
        if sections.get("skills") and len(sections["skills"]) >= 5: score += 20
        if sections.get("certifications") and len(sections["certifications"]) >= 1: score += 15
        
        # Bonus for depth/length of sections
        if resume_data.get("metrics", {}).get("experience_years", 0) >= 2: score += 10
        if resume_data.get("word_count", 0) > 400: score += 5

        return min(100, score)

    def score_keyword_relevance(self, resume_data):
        raw_text = resume_data.get("raw_text", "").lower()
        matched_keywords = 0
        keyword_counts = {}
        
        if not self.job_keywords:
            return 50

        # Expand keywords using Cohere if available
        expanded_keywords = self.expand_keywords_with_cohere(raw_text[:5000])
        
        for keyword in expanded_keywords:
            matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b|\b' + re.escape(keyword) + r'\s*\d*', raw_text, re.IGNORECASE))
            keyword_counts[keyword] = min(matches, 5)  # Cap at 5
            matched_keywords += keyword_counts[keyword]
        
        total_job_keywords = len(expanded_keywords)
        if total_job_keywords == 0:
            return 0
        
        relevance_percentage = (matched_keywords / total_job_keywords) * 100
        return min(100, int(relevance_percentage))

    def score_experience_relevance(self, resume_data):
        metrics = resume_data.get("metrics", {})
        sections = resume_data.get("sections", {})
        score = 0
        
        # Score based on years of experience
        experience_years = metrics.get("experience_years", 0)
        seniority = metrics.get("seniority", "Mid-Level")
        
        if seniority == "Senior Leader" and experience_years >= 8:
            score += 60
        elif experience_years >= 5:
            score += 40
        elif experience_years >= 2:
            score += 20
        else:
            score += 5
        
        # Score based on quantified achievements
        quantified_achievements = metrics.get("quantified_achievements", 0)
        if seniority != "Senior Leader":
            if quantified_achievements >= 3:
                score += 30
            elif quantified_achievements >= 1:
                score += 15
        else:
            score += 10

        # Score based on action verbs
        action_verbs = metrics.get("action_verbs", 0)
        if action_verbs >= 15:
            score += 20
        elif action_verbs >= 5:
            score += 10

        # Check if experience descriptions are substantial
        if sections.get("experience"):
            total_desc_length = sum(len(job) for job in sections["experience"])
            if total_desc_length > 500:
                score += 10

        return min(100, score)

    def score_education_match(self, resume_data):
        sections = resume_data.get("sections", {})
        education_entries = sections.get("education", [])
        score = 0

        relevant_degree_keywords = ['bachelor', 'master', 'phd', 'computer science', 'information technology', 'engineering', 'science', 'management information systems']
        
        if not education_entries:
            return 0

        for entry in education_entries:
            entry_lower = entry.lower()
            if any(kw in entry_lower for kw in relevant_degree_keywords):
                score += 50
            if re.search(r'(?:bachelor|master|phd)', entry_lower):
                score += 30
            if 'university' in entry_lower or 'institute' in entry_lower:
                score += 20
        
        # Add prestige bonus
        score += self.score_education_prestige(education_entries)
            
        return min(100, score)

    def score_skills_depth(self, resume_data):
        sections = resume_data.get("sections", {})
        skills = sections.get("skills", [])
        certifications = sections.get("certifications", [])
        score = 0

        if not skills:
            return 0

        if len(skills) >= 20: score += 40
        elif len(skills) >= 10: score += 20
        elif len(skills) >= 5: score += 10

        matched_job_skills = 0
        for skill in skills:
            if skill.lower() in self.job_keywords:
                matched_job_skills += 1
        
        if matched_job_skills > 0:
            score += min(40, matched_job_skills * 5)

        # Add certification tier score
        score += self.score_certification_tier(certifications)

        # Add skill cluster bonus
        score += self.score_skills_clusters(skills)

        return min(100, score)
    
    def _check_ats_issues(self, text: str) -> Dict[str, bool]:
        """Check for ATS-unfriendly elements."""
        issues = {
            "excessive_caps": bool(re.search(r'\b[A-Z]{4,}\b', text)),
            "tables_present": bool(re.search(r'\b(table|text box)\b', text, re.IGNORECASE)),
            "graphics_present": bool(re.search(r'\[IMAGE\]|●|♦|♣|♥|♠', text)),
            "font_inconsistency": bool(re.search(r'[A-Za-z]\s[A-Za-z]\s[A-Za-z]', text))  # Simple heuristic
        }
        return issues

    def _check_grammar_issues(self, text: str) -> Tuple[int, List[str]]:
        """Check for grammar/spelling issues using language_tool."""
        try:
            matches = self.language_tool.check(text)
            error_count = len(matches)
            examples = [match.ruleId for match in matches[:3]]  # Sample of error types
            return error_count, examples
        except Exception as e:
            logger.error(f"LanguageTool check failed: {str(e)}")
            return 0, []

    def score_formatting(self, resume_data: Dict) -> float:
        """Enhanced formatting scoring with ATS checks."""
        raw_text = resume_data.get("raw_text", "")
        score = 0
        
        # Basic formatting checks
        if re.search(r'•|-|\*', raw_text): score += 20
        if resume_data.get("sections", {}).keys(): score += 20
        if re.search(r'\d{4}\s*-\s*(?:\d{4}|Present|Current)', raw_text, re.IGNORECASE): score += 15
        word_count = resume_data.get("word_count", 0)
        if 300 <= word_count <= 800: score += 15
        
        # Check for ATS issues
        ats_issues = self._check_ats_issues(raw_text)
        if not ats_issues["excessive_caps"]: score += 10
        if not ats_issues["tables_present"]: score += 10
        if not ats_issues["graphics_present"]: score += 10
        
        return min(100, score)

    def score_clarity(self, resume_data: Dict) -> float:
        """Enhanced clarity scoring with grammar checks."""
        raw_text = resume_data.get("raw_text", "")
        metrics = resume_data.get("metrics", {})
        
        # Initialize score
        score = 0
        
        # Readability scoring
        readability = flesch_kincaid_grade(raw_text)
        if readability <= 12:
            score += 20
        elif readability <= 15:
            score += 10
        
        # Existing clarity metrics
        action_verbs_count = metrics.get("action_verbs", 0)
        if action_verbs_count >= 20:
            score += 30
        elif action_verbs_count >= 10:
            score += 15
        
        quantified_achievements = metrics.get("quantified_achievements", 0)
        if quantified_achievements >= 2:
            score += 20
        elif quantified_achievements >= 1:
            score += 10
        
        # Grammar and spelling checks
        error_count, _ = self._check_grammar_issues(raw_text)
        word_count = resume_data.get("word_count", 1)
        error_rate = error_count / word_count
        
        if error_rate < 0.01:  # Less than 1% errors
            score += 30
        elif error_rate < 0.03:  # Less than 3% errors
            score += 15
        elif error_rate < 0.05:  # Less than 5% errors
            score += 5
        
        # Check for excessive capitalization
        ats_issues = self._check_ats_issues(raw_text)
        if not ats_issues["excessive_caps"]:
            score += 5
        
        return min(100, score)

    def score_contact_info(self, resume_data):
        contact = resume_data.get("contact", {})
        score = 0
        if contact.get("name") and contact["name"] != "Unknown Name": score += 25
        if contact.get("email"): score += 25
        if contact.get("phone"): score += 25
        if contact.get("linkedin"): score += 25
        return min(100, score)

    def score_tone_positivity(self, resume_data):
        raw_text = resume_data.get("raw_text", "")
        if not raw_text:
            return 0
        
        # VADER sentiment analysis
        vader_scores = self.sid.polarity_scores(raw_text)
        vader_compound = vader_scores['compound']
        vader_normalized = ((vader_compound + 1) / 2) * 100  # Normalize -1 to 1 to 0-100
        
        # Cohere tone classification
        cohere_score = 50  # Default neutral score if Cohere is unavailable
        if self.co:
            try:
                # Define examples for tone classification
                examples = [
                    cohere.ClassifyExample(text="Developed and led a team to achieve 20% revenue growth.", label="positive"),
                    cohere.ClassifyExample(text="Proven track record of optimizing systems and delivering results.", label="professional"),
                    cohere.ClassifyExample(text="Failed to meet project deadlines due to resource constraints.", label="negative"),
                    cohere.ClassifyExample(text="Responsible for system administration tasks.", label="neutral"),
                    cohere.ClassifyExample(text="Achieved certifications and improved team efficiency.", label="positive"),
                    cohere.ClassifyExample(text="Managed complex projects with tight deadlines.", label="professional"),
                    cohere.ClassifyExample(text="Encountered challenges in system integration.", label="negative"),
                    cohere.ClassifyExample(text="Performed routine maintenance and updates.", label="neutral")
                ]
                
                # Split text into smaller chunks to avoid API limits (max 10,000 characters)
                max_chunk_length = 5000
                chunks = [raw_text[i:i + max_chunk_length] for i in range(0, len(raw_text), max_chunk_length)]
                
                # Classify each chunk
                positive_confidence = 0
                professional_confidence = 0
                chunk_count = len(chunks)
                
                try:
                    for chunk in chunks:
                        response = self.co.classify(inputs=[chunk], examples=examples)
                        for classification in response.classifications:
                            for label, prediction in classification.labels.items():
                                if label == "positive":
                                    positive_confidence += prediction.confidence
                                elif label == "professional":
                                    professional_confidence += prediction.confidence
                except Exception as e:
                    logger.error(f"Cohere classify failed: {str(e)}")
                    cohere_score = 50
                
                # Average confidences across chunks
                if chunk_count > 0:
                    positive_confidence /= chunk_count
                    professional_confidence /= chunk_count
                
                # Combine positive and professional confidences (weighted)
                cohere_score = (positive_confidence * 0.6 + professional_confidence * 0.4) * 100
                logger.debug(f"Cohere tone scores: positive={positive_confidence}, professional={professional_confidence}, combined={cohere_score}")
            
            except Exception as e:
                logger.error(f"Cohere API error: {str(e)}")
                cohere_score = 50  # Fallback to neutral score
        
        # Combine VADER and Cohere scores (weighted average, 60% VADER, 40% Cohere)
        combined_score = (0.6 * vader_normalized + 0.4 * cohere_score)
        return min(100, round(combined_score, 2))

    def generate_tone_feedback(self, resume_data, tone_score):
        feedback = []
        raw_text = resume_data.get("raw_text", "")
        metrics = resume_data.get("metrics", {})
        
        if not raw_text:
            feedback.append("No text available for tone analysis. Please check if your resume was parsed correctly.")
            return feedback
        
        # VADER sentiment analysis
        vader_scores = self.sid.polarity_scores(raw_text)
        vader_compound = vader_scores['compound']
        
        # Cohere tone classification
        cohere_labels = {"positive": 0, "professional": 0, "neutral": 0, "negative": 0}
        if self.co:
            try:
                examples = [
                    cohere.ClassifyExample(text="Led a cross-functional team to deliver a 30% improvement in system efficiency.", label="positive"),
                    cohere.ClassifyExample(text="Designed and implemented scalable cloud infrastructure supporting 1M+ users.", label="professional"),
                    cohere.ClassifyExample(text="Project faced delays due to unclear requirements from stakeholders.", label="negative"),
                    cohere.ClassifyExample(text="Responsible for maintaining Windows Server 2016 environment.", label="neutral"),
                    cohere.ClassifyExample(text="Recognized with 'Employee of the Year' for automating critical processes.", label="positive"),
                    cohere.ClassifyExample(text="Certified AWS Solutions Architect with 5+ years of cloud migration experience.", label="professional"),
                    cohere.ClassifyExample(text="Struggled to align team members on project priorities.", label="negative"),
                    cohere.ClassifyExample(text="Performed monthly security patches and system updates.", label="neutral")
                ]

                # Find negative phrases and suggest alternatives
                negative_phrases = re.findall(r'\b(failed|problem|issue|challenge|difficult|struggle|limitation)\b', raw_text, re.IGNORECASE)
                if negative_phrases:
                    unique_phrases = list(set(negative_phrases[:3]))
                    replacements = {
                        'failed': 'learned valuable lessons from',
                        'problem': 'opportunity for improvement',
                        'issue': 'situation requiring attention',
                        'challenge': 'opportunity to demonstrate problem-solving',
                        'difficult': 'complex',
                        'struggle': 'worked diligently to overcome',
                        'limitation': 'parameter'
                    }
                    suggestion = "Consider replacing phrases like '{}' with more constructive alternatives such as '{}'.".format(
                        "', '".join(unique_phrases),
                        "', '".join([replacements.get(phrase.lower(), phrase) for phrase in unique_phrases])
                    )
                    feedback.append(suggestion)
                
                # Analyze text in chunks
                max_chunk_length = 5000
                chunks = [raw_text[i:i + max_chunk_length] for i in range(0, len(raw_text), max_chunk_length)]
                
                for chunk in chunks:
                    response = self.co.classify(inputs=[chunk], examples=examples)
                    for classification in response.classifications:
                        for label, prediction in classification.labels.items():
                            cohere_labels[label] += prediction.confidence
                
                # Average confidences
                chunk_count = len(chunks)
                if chunk_count > 0:
                    for label in cohere_labels:
                        cohere_labels[label] /= chunk_count
                
                logger.debug(f"Cohere tone confidences: {cohere_labels}")
            
            except Exception as e:
                logger.error(f"Cohere API error in feedback: {str(e)}")
                cohere_labels = {"positive": 0.25, "professional": 0.25, "neutral": 0.25, "negative": 0.25}
        
        # Generate feedback based on combined tone score
        if tone_score >= 80:
            feedback.append("Excellent tone! Your resume effectively balances professionalism with enthusiasm. Maintain this approach while ensuring all sections are equally strong.")
        elif tone_score >= 60:
            feedback.append(
                "Good overall tone. To improve, consider these rewrites:\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][0]['before']} → {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][0]['after']}\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['passive_phrases'][0]['before']} → {self.EXAMPLE_TRANSFORMATIONS['passive_phrases'][0]['after']}"
            )
        elif tone_score >= 40:
            feedback.append(
                "Your resume tone needs more energy. Try these transformations:\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][1]['before']} → {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][1]['after']}\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['negative_phrases'][0]['before']} → {self.EXAMPLE_TRANSFORMATIONS['negative_phrases'][0]['after']}"
            )
        else:
            feedback.append(
                "Significant tone improvement needed. Use these examples:\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][2]['before']} → {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][2]['after']}\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['negative_phrases'][1]['before']} → {self.EXAMPLE_TRANSFORMATIONS['negative_phrases'][1]['after']}\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['passive_phrases'][1]['before']} → {self.EXAMPLE_TRANSFORMATIONS['passive_phrases'][1]['after']}"
            )
        
        # VADER-specific feedback
        if vader_scores['neg'] > 0.1:
            feedback.append("Negative language detected. Instead of 'Failed to meet deadline', try 'Implemented process improvements to ensure future deadlines were met'.")
        if vader_scores['pos'] < 0.2:
            positive_verbs = ["achieved", "transformed", "optimized", "accelerated", "mentored"]
            feedback.append(f"Incorporate more positive language. Try verbs like: {', '.join(positive_verbs)}. Example: 'Optimized server configuration, reducing downtime by 25%'.")
        if vader_compound < 0.4:
            feedback.append("Boost enthusiasm by highlighting what excited you about projects. Example: 'Passionately led migration to cloud infrastructure, completing project 2 weeks early'.")
        
        # Cohere-specific feedback
        if cohere_labels['negative'] > 0.3:
            feedback.append("Some sections read negatively. For challenges faced, emphasize solutions: 'Identified system vulnerability and implemented patching protocol that became company standard'.")
        if cohere_labels['professional'] < 0.3:
            feedback.append("Increase professional impact by: 1) Listing relevant certifications, 2) Using industry-standard terminology, 3) Including brief case studies of complex problems solved.")
        if cohere_labels['neutral'] > 0.5:
            feedback.append("Too many neutral statements. Transform 'Responsible for network administration' to 'Redesigned network infrastructure supporting 500+ users, improving reliability by 40%'.")
        if cohere_labels['positive'] < 0.3:
            feedback.append("Highlight achievements more prominently. For each position, include: 1) Business impact, 2) Technical complexity overcome, 3) Recognition received.")
        
        # Additional tone feedback based on metrics
        action_verbs = metrics.get("action_verbs", 0)
        if action_verbs < 10:
            strong_verbs = ["Architected", "Championed", "Engineered", "Pioneered", "Streamlined"]
            feedback.append(
                f"Use more dynamic verbs. Example:\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][0]['before']} → {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][0]['after']}\n"
                f"Strong verbs to consider: {', '.join(strong_verbs)}."
            )
        
        quantified_achievements = metrics.get("quantified_achievements", 0)
        if quantified_achievements < 2:
            feedback.append("Quantify more achievements. Examples: 'Reduced server costs 30% through virtualization', 'Cut ticket resolution time from 48 to 12 hours'.")
        
        # Check for overly formal language
        formal_phrases = re.findall(r'\b(hereby|henceforth|tasked with|undertake|utilize|amongst)\b', raw_text, re.IGNORECASE)
        if formal_phrases:
            simpler_alternatives = {
                'hereby': 'here',
                'henceforth': 'from now on',
                'tasked with': 'responsible for',
                'undertake': 'take on',
                'utilize': 'use',
                'amongst': 'among'
            }
            feedback.append(f"Simplify formal language. Try: {', '.join([f'\"{phrase}\" → \"{simpler_alternatives.get(phrase.lower(), phrase)}\"' for phrase in formal_phrases[:3]])}")
        
        return feedback

    def generate_feedback(self, resume_data: Dict, score_breakdown_weighted: Dict) -> List[str]:
        feedback = []
        raw_text = resume_data.get("raw_text", "")
        contact = resume_data.get("contact", {})
        sections = resume_data.get("sections", {})
        metrics = resume_data.get("metrics", {})
        seniority = metrics.get("seniority", "Mid-Level")

        # Calculate word count accurately
        word_count = len(raw_text.split()) if raw_text else 0
        metrics["word_count"] = word_count  # Update metrics for consistency

        # Contact info feedback
        if contact.get("name") == "Unknown Name":
            feedback.append("🔍 Name not detected. Place your full name prominently at the top in 18-22pt font.")
        if not contact.get("email"):
            feedback.append("✉️ Add a professional email (first.last@domain.com format preferred).")
        if not contact.get("phone"):
            feedback.append("📞 Include a phone number with country code if applying internationally.")
        if not contact.get("linkedin"):
            feedback.append("🔗 Add LinkedIn profile (customize URL to include your name).")

        # Section-specific feedback
        if not sections.get("summary") or len(sections["summary"].strip()) < 50:
            feedback.append("📝 Write a 3-4 line professional summary highlighting: 1) Your expertise, 2) Years of experience, 3) Key achievements.")
        
        if len(sections.get("experience", [])) < 1:
            feedback.append("💼 Add work experience with: Job title, Company, Dates, and 3-5 bullet points per role focusing on achievements.")
        elif seniority in ["Senior", "Manager"]:
            if metrics.get("quantified_achievements", 0) < 3:
                feedback.append("📊 Strengthen experience by adding more quantified achievements (e.g., 'reduced network downtime by 30%'). Aim for 3-5 per role.")
            if metrics.get("action_verbs", 0) < 15:
                feedback.append("⚡ Use more dynamic action verbs (e.g., 'architected', 'streamlined') to highlight leadership. Aim for 15+ across roles.")
        
        if len(sections.get("education", [])) < 1:
            feedback.append("🎓 Include education: Degree, Institution, Graduation year. Add GPA if above 3.5 (optional for senior roles).")
        elif seniority in ["Senior", "Manager"]:
            feedback.append("🎓 For senior roles, prioritize recent certifications or advanced degrees over GPA unless exceptional.")

        # Skills and certifications feedback
        if len(sections.get("skills", [])) < 5:
            feedback.append("🛠️ List 8-12 technical skills. Group them (e.g., 'Cloud: AWS, Azure, GCP') for better readability.")
        elif len(sections.get("skills", [])) < 10 and seniority in ["Senior", "Manager"]:
            feedback.append("🛠️ Expand skills section to 10-15 technical skills, organized by category (e.g., 'Networking: Cisco, VPN', 'Cloud: AWS, Azure').")
        
        if not sections.get("certifications"):
            feedback.append("🏆 Add relevant certifications (e.g., CISSP, CCNP, AWS Certified Solutions Architect) to boost credibility.")
        elif len(sections.get("certifications", [])) < 2 and seniority in ["Senior", "Manager"]:
            feedback.append("🏆 List 2+ high-tier certifications (e.g., CISSP, CCNP) to align with senior IT roles.")

        # Action verbs feedback with examples
        action_verbs = resume_data.get("metrics", {}).get("action_verbs", 0)
        if action_verbs < 15 and resume_data.get("metrics", {}).get("seniority", "Mid-Level") in ["Senior", "Manager", "Senior Leader"]:
            feedback.append(
                f"⚡ Boost leadership impact with stronger verbs. Example:\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][1]['before']} → {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][1]['after']}"
            )    

        # ATS optimization feedback
        ats_issues = self._check_ats_issues(raw_text)
        if ats_issues["excessive_caps"]:
            # Only flag excessive caps if not standard headers
            headers = re.findall(r'^[A-Z\s]{10,}$', raw_text, re.MULTILINE)
            if headers and all(len(h.split()) <= 3 for h in headers):  # Allow short headers like "EXPERIENCE"
                pass
            else:
                feedback.append("🔠 Reduce excessive ALL CAPS in body text. Use title case for headers.")
        if ats_issues["tables_present"]:
            feedback.append("📊 Avoid tables. Use simple bullet points for better ATS parsing.")
        if ats_issues["graphics_present"]:
            feedback.append("🖼️ Replace graphic bullets (e.g., ●) with standard text bullets (-) for ATS compatibility.")
        if ats_issues["font_inconsistency"]:
            feedback.append("✒️ Use 1-2 professional fonts consistently (e.g., Calibri or Arial).")

        # Quantified achievements feedback with examples
        quantified_achievements = resume_data.get("metrics", {}).get("quantified_achievements", 0)
        if quantified_achievements < 2:
            feedback.append(
                f"📊 Quantify achievements. Example:\n"
                f"- {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][2]['before']} → {self.EXAMPLE_TRANSFORMATIONS['weak_verbs'][2]['after']}"
            )    

        # Grammar and clarity feedback
        error_count, error_examples = self._check_grammar_issues(raw_text)
        error_rate = (error_count / max(word_count, 1)) * 100 if word_count > 0 else 0
        if error_count > 0:
            common_errors = {
                'EN_CONTRACTION_SPELLING': "Avoid contractions (use 'do not' instead of 'don't')",
                'MORFOLOGIK_RULE_EN_US': "Spelling error",
                'UPPERCASE_SENTENCE_START': "Sentence doesn't start with capital letter",
                'ENGLISH_WORD_REPEAT_BEGINNING_RULE': "Repeated word at sentence start"
            }
            examples_with_explanations = [
                f"{example} ({common_errors.get(example, 'grammar issue')})"
                for example in error_examples
            ]
            if error_rate > 5:  # Only report significant error rates
                feedback.append(
                    f"✏️ Found {error_count} grammar/spelling issues (~{error_rate:.1f}% of text). "
                    f"Examples: {'; '.join(examples_with_explanations[:3])}. "
                    "Use a grammar checker like Grammarly for detailed corrections."
                )
            elif error_count > 0:
                feedback.append(
                    f"✏️ Minor grammar/spelling issues detected ({error_count}). Examples: {'; '.join(examples_with_explanations[:3])}. Review with a grammar checker."
                )

        # Formatting suggestions
        if not re.search(r'•|-|\*', raw_text):
            feedback.append("🔘 Use bullet points (• or -) instead of paragraphs for experience descriptions.")
        
        if not re.search(r'\d{4}\s*-\s*(?:\d{4}|Present|Current)', raw_text, re.IGNORECASE):
            feedback.append("📅 Format dates consistently (e.g., '03/2020 - Present' or 'January 2020 - Current').")

        # Length feedback
        if word_count > 1000:
            feedback.append("📏 Resume is too long (>1000 words). Trim by: 1) Summarizing roles older than 10 years, 2) Combining similar bullet points, 3) Focusing on recent achievements.")
        elif word_count < 400 and seniority in ["Senior", "Manager"]:
            feedback.append("📏 Resume is short for a senior role (<400 words). Expand by: 1) Adding metrics to achievements, 2) Detailing leadership roles, 3) Including recent certifications.")

        # Keyword relevance feedback
        if score_breakdown_weighted.get("keyword_relevance", 0) < 15:  # Assuming weight=25, 60% threshold
            feedback.append("🔑 Add more job-specific keywords (e.g., 'network security', 'cloud infrastructure') from the job description to improve ATS match.")
        
        # Certifications feedback
        certifications = resume_data.get("sections", {}).get("certifications", [])
        outdated_certs = [cert for cert in certifications if any(oc in cert.lower() for oc in self.CERTIFICATION_TIERS["Outdated"])]
        if outdated_certs:
            feedback.append(
                f"📜 Outdated certifications detected ({', '.join(outdated_certs[:2])}). "
                "Consider replacing with modern equivalents like AWS Certified Cloud Practitioner or CompTIA Security+."
            )

        # Positive reinforcement for strong resumes
        if len(feedback) < 5 and error_count < 3 and 400 <= word_count <= 1000:
            feedback.append("🌟 Strong resume! To stand out: 1) Add 1-2 high-impact metrics, 2) Include a projects section for complex IT initiatives, 3) Tailor skills to specific job postings.")

        return feedback if feedback else ["🌟 Excellent resume! Tailor keywords slightly for each job to maximize ATS compatibility."]