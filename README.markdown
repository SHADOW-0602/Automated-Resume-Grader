# Resume Grader

Resume Grader is a web-based application that analyzes resumes in PDF, DOCX, or TXT formats, providing a score out of 100, personalized improvement suggestions, and tone analysis. It leverages NLP techniques and ATS (Applicant Tracking System) compatibility checks to help job seekers optimize their resumes for IT and tech-related roles. The application also tracks version history for iterative resume improvements.

## Features

- **Resume Scoring**: Evaluates resumes based on section completeness, keyword relevance, experience, education, skills, formatting, clarity, contact info, and tone positivity.
- **Personalized Feedback**: Offers actionable suggestions to improve resume content and ATS compatibility.
- **Tone Analysis**: Uses VADER sentiment analysis and Cohere AI to assess the resume's tone, suggesting ways to make it more professional and positive.
- **Version History**: Tracks multiple resume uploads for the same job title, allowing users to compare versions and monitor improvements.
- **ATS Optimization**: Detects ATS-unfriendly elements like tables, graphics, or excessive capitalization.
- **Responsive UI**: Built with HTML, CSS (Tailwind-inspired), and JavaScript for a modern, user-friendly experience.
- **Backend Processing**: Powered by Flask, MongoDB for storage, and libraries like pdfplumber, PyMuPDF, and spaCy for resume parsing.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Database**: MongoDB
- **NLP Libraries**: spaCy, NLTK, Cohere, language_tool_python, textstat
- **File Processing**: pdfplumber, PyMuPDF, python-docx, pytesseract
- **Environment**: dotenv for configuration, logging for debugging

## Prerequisites

- Python 3.8+
- MongoDB (local or cloud instance, e.g., MongoDB Atlas)
- Tesseract OCR (for scanned PDFs)
- Node.js (optional, for frontend development)
- Cohere API key (for advanced NLP features)
- A modern web browser

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/resume-grader.git
   cd resume-grader
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Set Up Environment Variables**
   Create a `.env` file in the root directory:
   ```plaintext
   MONGO_URI=mongodb://localhost:27017/resume_grader
   COHERE_API_KEY=your_cohere_api_key
   ```

6. **Install spaCy Model**
   ```bash
   python -m spacy download en_core_web_lg
   ```

7. **Run the Application**
   ```bash
   python backend/app.py
   ```
   The app will be available at `http://localhost:8000`.

## Usage

1. **Access the Web Interface**
   Open `http://localhost:8000` in your browser.

2. **Upload a Resume**
   - Drag and drop or browse to upload a resume (PDF, DOCX, or TXT) from the frontend interface.
   - Optionally, enter a target job title to tailor keyword analysis.
   - Click "Grade My Resume" to process.

3. **View Results**
   - **Score**: A score out of 100 based on multiple criteria.
   - **Feedback**: Actionable suggestions to improve content, formatting, and ATS compatibility.
   - **Tone Analysis**: Insights into the resume's tone with suggestions for positivity and professionalism.
   - **Version History**: Review past uploads for the same job title.

4. **Iterate**
   Update your resume based on feedback, re-upload, and track improvements via version history.

## Project Structure

```plaintext
resume-grader/
├── backend/
│   ├── app.py
│   ├── parser.py
│   └── scorer.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── images/
│   └── favicon.ico
├── uploads/
├── requirements.txt
├── .env
└── README.md
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/), [MongoDB](https://www.mongodb.com/), and [Cohere](https://cohere.ai/).
- Inspired by the need for job seekers to optimize resumes for ATS and hiring managers.
- Icons from [Font Awesome](https://fontawesome.com/).

## Contact

For questions or feedback, open an issue on GitHub or reach out via [your.email@example.com](mailto:your.email@example.com).