const CONFIG = {
  API_BASE_URL: 'http://localhost:8000/api',
  ALLOWED_EXTENSIONS: ['pdf', 'docx', 'txt'],
  SCORE_MESSAGES: {
    90: "Excellent! Your resume is well-optimized and professional.",
    80: "Great job! Your resume is strong but could use minor improvements.",
    70: "Good start! Some areas need refinement to stand out.",
    60: "Fair. Several improvements needed to be competitive.",
    50: "Needs work. Significant revisions recommended.",
    0: "Poor. Consider a complete rewrite with professional guidance."
  }
};

/** DOM Elements */
const elements = {
  uploadForm: document.getElementById('uploadForm'),
  resumeUpload: document.getElementById('resumeUpload'),
  jobTitle: document.getElementById('jobTitle'),
  submitBtn: document.getElementById('submitBtn'),
  loadingDiv: document.getElementById('loading'),
  resultsDiv: document.getElementById('results'),
  scoreValue: document.getElementById('scoreValue'),
  scoreNumber: document.querySelector('.score-number'),
  scoreMessage: document.getElementById('scoreMessage'),
  feedbackList: document.getElementById('feedbackList'),
  toneAnalysis: document.getElementById('toneAnalysis'),
  viewHistoryBtn: document.getElementById('viewHistoryBtn'),
  versionHistory: document.getElementById('versionHistory'),
  versionList: document.getElementById('versionList'),
  fileDisplay: document.getElementById('fileDisplay'),
};

const handleFileSelect = (file) => {
  if (!file) {
    elements.fileDisplay.textContent = '';
    elements.fileDisplay.classList.add('empty');
    return;
  }

  elements.fileDisplay.innerHTML = `
    <i class="fas fa-file-alt"></i>
    <span>${file.name}</span>
    <small>(${(file.size / 1024).toFixed(1)} KB)</small>
  `;
  elements.fileDisplay.classList.remove('empty');
};

const setupDragAndDrop = () => {
  const uploadWrapper = document.querySelector('.file-upload-wrapper');
  const fileInput = elements.resumeUpload;

  // Handle drag over
  uploadWrapper.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadWrapper.classList.add('drag-over');
  });

  // Handle drag leave
  uploadWrapper.addEventListener('dragleave', () => {
    uploadWrapper.classList.remove('drag-over');
  });

  // Handle drop
  uploadWrapper.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadWrapper.classList.remove('drag-over');
    
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  // Handle file input change
  fileInput.addEventListener('change', () => {
    handleFileSelect(fileInput.files[0]);
  });
};

/**
 * Displays an error message to the user.
 * @param {string} message - The error message to display.
 */
const showError = (message) => {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-message';
  errorDiv.innerHTML = `
    <i class="fas fa-exclamation-circle"></i>
    <span>${message}</span>
  `;
  
  // Insert error message before the form
  elements.uploadForm.parentNode.insertBefore(errorDiv, elements.uploadForm);
  
  // Remove error after 5 seconds
  setTimeout(() => {
    errorDiv.classList.add('fade-out');
    setTimeout(() => errorDiv.remove(), 300);
  }, 5000);
  
  elements.submitBtn.disabled = false;
  elements.loadingDiv.style.display = 'none';
};

/**
 * Gets appropriate score message based on the score.
 * @param {number} score - The resume score.
 * @returns {string} - The score message.
 */
const getScoreMessage = (score) => {
  for (const [threshold, message] of Object.entries(CONFIG.SCORE_MESSAGES).sort((a, b) => b[0] - a[0])) {
    if (score >= parseInt(threshold)) {
      return message;
    }
  }
  return "";
};

/**
 * Validates the selected file.
 * @param {File} file - The selected file.
 * @returns {Object} - Validation result { isValid: boolean, error: string }
 */
const validateFile = (file) => {
  if (!file) {
    showError('Please select a resume file.');
    return { isValid: false, error: 'Please select a resume file.' };
  }
  
  const extension = file.name.split('.').pop().toLowerCase();
  if (!CONFIG.ALLOWED_EXTENSIONS.includes(extension)) {
    showError(`Unsupported file type. Please upload a ${CONFIG.ALLOWED_EXTENSIONS.join(', ').toUpperCase()} file.`);
    return {
      isValid: false,
      error: `Unsupported file type. Please upload a ${CONFIG.ALLOWED_EXTENSIONS.join(', ').toUpperCase()} file.`,
    };
  }
  
  return { isValid: true, error: '' };
};

/**
 * Uploads the resume and processes the server response.
 * @param {Event} e - Form submit event.
 */
const handleUpload = async (e) => {
  e.preventDefault();

  const file = elements.resumeUpload.files[0];
  const validation = validateFile(file);
  if (!validation.isValid) {
    showError(validation.error);
    return;
  }

  elements.loadingDiv.style.display = 'block';
  elements.resultsDiv.style.display = 'none';
  elements.submitBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', file);
    if (elements.jobTitle.value.trim()) {
      formData.append('job_title', elements.jobTitle.value.trim());
    }

    const response = await fetch(`${CONFIG.API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    if (!data.success) {
      throw new Error(data.error || 'Unknown server error');
    }

    displayResults(data);
  } catch (error) {
    showError(error.message);
  } finally {
    elements.loadingDiv.style.display = 'none';
    elements.submitBtn.disabled = false;
  }
};

/**
 * Displays the resume analysis results.
 * @param {Object} data - Server response data.
 */
const displayResults = (data) => {
  // Update score with animation
  animateValue(elements.scoreNumber, 0, data.score, 1000);
  
  // Update score message
  elements.scoreMessage.textContent = getScoreMessage(data.score);
  
  // Update feedback
  elements.feedbackList.innerHTML = data.feedback
    .map((item) => `<li>${item}</li>`)
    .join('');
  
  // Update tone analysis
  elements.toneAnalysis.innerHTML = data.tone_analysis
    .map((item) => `<li>${item}</li>`)
    .join('');
  
  // Configure version history button
  elements.viewHistoryBtn.style.display = 'block';
  elements.viewHistoryBtn.dataset.resumeGroupId = data.resume_group_id;
  elements.versionHistory.style.display = 'none';
  elements.versionList.innerHTML = '';
  
  // Show results section
  elements.resultsDiv.style.display = 'block';
  elements.resultsDiv.scrollIntoView({ behavior: 'smooth' });
};

/**
 * Animates a value from start to end.
 * @param {HTMLElement} element - The element to animate.
 * @param {number} start - Start value.
 * @param {number} end - End value.
 * @param {number} duration - Animation duration in ms.
 */
const animateValue = (element, start, end, duration) => {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    element.textContent = Math.floor(progress * (end - start) + start);
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
};

/**
 * Fetches and displays version history for a resume group.
 */
const handleViewHistory = async () => {
  const resumeGroupId = elements.viewHistoryBtn.dataset.resumeGroupId;
  if (!resumeGroupId) {
    showError('No version history available.');
    return;
  }

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/history/${resumeGroupId}`);
    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    if (!data.success) {
      throw new Error(data.error || 'Unknown server error');
    }

    displayVersionHistory(data.versions);
  } catch (error) {
    showError(`Failed to fetch version history: ${error.message}`);
  }
};

/**
 * Displays version history in the UI.
 * @param {Array} versions - Array of version objects.
 */
const displayVersionHistory = (versions) => {
  elements.versionList.innerHTML = versions
    .map((version) => {
      const uploadDate = new Date(version.upload_date).toLocaleString();
      return `
        <li>
          <div class="version-summary">
            <strong>Version ${version.version_number}</strong> - Score: ${version.score} - Uploaded: ${uploadDate}
            <span class="version-toggle" data-file-id="${version.file_id}">Show Details</span>
          </div>
          <div class="version-details" id="details-${version.file_id}" style="display: none;">
            <strong>Feedback:</strong>
            <ul>${version.feedback.map((item) => `<li>${item}</li>`).join('')}</ul>
            <strong>Tone Analysis:</strong>
            <ul>${version.tone_analysis.map((item) => `<li>${item}</li>`).join('')}</ul>
          </div>
        </li>
      `;
    })
    .join('');

  elements.versionHistory.style.display = 'block';
  elements.versionHistory.scrollIntoView({ behavior: 'smooth' });

  // Attach toggle event listeners
  document.querySelectorAll('.version-toggle').forEach((toggle) => {
    toggle.addEventListener('click', () => {
      const fileId = toggle.dataset.fileId;
      const details = document.getElementById(`details-${fileId}`);
      const isHidden = details.style.display === 'none' || !details.style.display;
      details.style.display = isHidden ? 'block' : 'none';
      toggle.textContent = isHidden ? 'Hide Details' : 'Show Details';
    });
  });
};

/**
 * Initializes event listeners.
 */
const init = () => {
  elements.uploadForm.addEventListener('submit', handleUpload);
  elements.viewHistoryBtn.addEventListener('click', handleViewHistory);
  setupDragAndDrop();
};

// Start the application
document.addEventListener('DOMContentLoaded', init);