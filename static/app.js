class PalmDiseaseDetector {
    constructor() {
        this.selectedFile = null;
        this.currentAnalysis = null;
        this.initializeTheme();
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeTheme() {
        // Initialize theme from localStorage or system preference
        const savedTheme = localStorage.getItem('theme');
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const theme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
        
        document.documentElement.setAttribute('data-theme', theme);
        this.currentTheme = theme;
    }

    initializeElements() {
        // File upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.uploadPrompt = document.getElementById('upload-prompt');
        this.uploadPreview = document.getElementById('upload-preview');
        this.previewImage = document.getElementById('preview-image');
        this.fileName = document.getElementById('file-name');
        this.browseBtn = document.getElementById('browse-btn');
        this.changeImageBtn = document.getElementById('change-image');
        
        // Action buttons
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.helpBtn = document.getElementById('help-btn');
        this.retryBtn = document.getElementById('retry-btn');
        this.newAnalysisBtn = document.getElementById('new-analysis');
        this.downloadReportBtn = document.getElementById('download-report');
        this.themeToggle = document.getElementById('theme-toggle');
        
        // Display panels
        this.infoPanel = document.getElementById('info-panel');
        this.loadingPanel = document.getElementById('loading-panel');
        this.resultsSection = document.getElementById('results-section');
        this.errorSection = document.getElementById('error-section');
        this.progressBar = document.getElementById('progress-bar');
        this.progressText = document.getElementById('progress-text');
        
        // Results elements
        this.resultImage = document.getElementById('result-image');
        this.diseaseName = document.getElementById('disease-name');
        this.confidenceScore = document.getElementById('confidence-score');
        this.diseaseDescription = document.getElementById('disease-description');
        this.symptomsList = document.getElementById('symptoms-list');
        this.treatmentRecommendations = document.getElementById('treatment-recommendations');
        this.alternativeDiseases = document.getElementById('alternative-diseases');
        this.primaryResult = document.getElementById('primary-result');
        this.errorMessage = document.getElementById('error-message');
    }

    attachEventListeners() {
        // File upload events
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.changeImageBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadArea.addEventListener('click', () => {
            if (!this.selectedFile) this.fileInput.click();
        });
        
        // Action button events
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        this.clearBtn.addEventListener('click', () => this.clearSelection());
        this.helpBtn.addEventListener('click', () => this.showHelp());
        this.retryBtn.addEventListener('click', () => this.hideError());
        this.newAnalysisBtn.addEventListener('click', () => this.resetForNewAnalysis());
        this.downloadReportBtn.addEventListener('click', () => this.downloadReport());
        this.themeToggle.addEventListener('click', () => this.toggleTheme());
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Invalid file type. Please upload JPG or PNG images only.');
            return;
        }

        // Validate file size (10MB max)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File too large. Maximum size is 10MB.');
            return;
        }

        this.selectedFile = file;
        this.displayPreview(file);
        this.analyzeBtn.disabled = false;
        this.clearBtn.classList.remove('hidden');
    }

    displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.fileName.textContent = file.name;
            this.uploadPrompt.classList.add('hidden');
            this.uploadPreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    clearSelection() {
        this.selectedFile = null;
        this.fileInput.value = '';
        this.uploadPrompt.classList.remove('hidden');
        this.uploadPreview.classList.add('hidden');
        this.analyzeBtn.disabled = true;
        this.clearBtn.classList.add('hidden');
        this.hideResults();
        this.hideError();
        this.showInfoPanel();
    }

    async analyzeImage() {
        if (!this.selectedFile) return;

        this.showLoadingPanel();
        this.hideResults();
        this.hideError();

        try {
            const formData = new FormData();
            formData.append('image', this.selectedFile);

            // Simulate progress
            this.updateProgress(0, 'Uploading image...');
            await this.delay(500);
            
            this.updateProgress(30, 'Preprocessing image...');
            await this.delay(500);
            
            this.updateProgress(60, 'Running AI analysis...');
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            this.updateProgress(90, 'Processing results...');
            await this.delay(300);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Analysis failed');
            }

            const result = await response.json();
            this.updateProgress(100, 'Complete!');
            
            await this.delay(500);
            this.currentAnalysis = result;
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'An error occurred during analysis. Please try again.');
        } finally {
            this.hideLoadingPanel();
        }
    }

    updateProgress(percentage, text) {
        this.progressBar.style.width = `${percentage}%`;
        this.progressText.textContent = text;
    }

    displayResults(result) {
        const diseaseInfo = result.disease_info;
        
        // Set result image
        this.resultImage.src = this.previewImage.src;
        
        // Set disease name
        this.diseaseName.textContent = diseaseInfo.name;
        
        // Set confidence score with appropriate styling
        const confidence = Math.round(result.confidence * 100);
        this.confidenceScore.textContent = `${confidence}% Confidence`;
        this.confidenceScore.className = this.getConfidenceClass(confidence);
        
        // Set description
        this.diseaseDescription.textContent = diseaseInfo.description;
        
        // Set primary result background based on severity
        this.primaryResult.className = `mb-6 p-4 rounded-lg ${this.getSeverityClass(diseaseInfo.severity)}`;
        
        // Populate symptoms
        this.populateList(this.symptomsList, diseaseInfo.symptoms);
        
        // Populate treatment recommendations
        this.populateList(this.treatmentRecommendations, diseaseInfo.treatment);
        
        // Populate alternative possibilities
        this.populateAlternatives(result.alternatives);
        
        // Show results with animation
        this.resultsSection.classList.remove('hidden');
        this.resultsSection.classList.add('fade-in');
        
        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    populateList(container, items) {
        container.innerHTML = '';
        items.forEach(item => {
            const li = document.createElement('li');
            li.className = 'flex items-start';
            li.innerHTML = `
                <i class="fas fa-check-circle text-green-600 mt-1 mr-2"></i>
                <span class="text-gray-700 dark:text-gray-300">${item}</span>
            `;
            container.appendChild(li);
        });
    }

    populateAlternatives(alternatives) {
        this.alternativeDiseases.innerHTML = '';
        
        // Skip the primary result and show next alternatives
        alternatives.slice(1, 4).forEach(alt => {
            const confidence = Math.round(alt.confidence * 100);
            const div = document.createElement('div');
            div.className = 'alternative-disease-card';
            div.innerHTML = `
                <div class="flex justify-between items-center">
                    <span class="font-medium text-gray-900 dark:text-gray-100">${this.formatDiseaseName(alt.disease)}</span>
                    <span class="text-xs px-2 py-1 rounded-full ${this.getConfidenceClass(confidence)}">${confidence}%</span>
                </div>
            `;
            this.alternativeDiseases.appendChild(div);
        });
    }

    formatDiseaseName(disease) {
        return disease.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    getConfidenceClass(confidence) {
        if (confidence >= 80) return 'confidence-high bg-green-600 text-white text-sm px-3 py-1 rounded-full';
        if (confidence >= 60) return 'confidence-medium bg-yellow-500 text-black text-sm px-3 py-1 rounded-full';
        return 'confidence-low bg-red-600 text-white text-sm px-3 py-1 rounded-full';
    }

    getSeverityClass(severity) {
        switch (severity.toLowerCase()) {
            case 'none': return 'severity-none bg-green-50 border border-green-200';
            case 'moderate': return 'severity-moderate bg-yellow-50 border border-yellow-200';
            case 'severe': return 'severity-severe bg-red-50 border border-red-200';
            default: return 'severity-unknown bg-gray-50 border border-gray-200';
        }
    }

    showLoadingPanel() {
        this.infoPanel.classList.add('hidden');
        this.loadingPanel.classList.remove('hidden');
        this.progressBar.style.width = '0%';
    }

    hideLoadingPanel() {
        this.loadingPanel.classList.add('hidden');
        this.showInfoPanel();
    }

    showInfoPanel() {
        this.infoPanel.classList.remove('hidden');
    }

    showResults() {
        this.resultsSection.classList.remove('hidden');
    }

    hideResults() {
        this.resultsSection.classList.add('hidden');
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorSection.classList.remove('hidden');
        this.errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideError() {
        this.errorSection.classList.add('hidden');
    }

    resetForNewAnalysis() {
        this.clearSelection();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    showHelp() {
        alert(`Palm Tree Disease Detector Help

How to use:
1. Upload a clear, well-lit photo of your palm tree
2. Click 'Analyze Image' to run AI analysis
3. Review the results and recommendations

Tips for best results:
• Take photos in natural daylight
• Include both healthy and diseased areas
• Ensure the image is in focus
• Use JPG or PNG format, max 10MB

Note: This tool provides guidance only. For serious plant health concerns, consult a professional arborist or plant pathologist.`);
    }

    downloadReport() {
        if (!this.currentAnalysis) return;

        const report = this.generateReport();
        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `palm-disease-analysis-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    generateReport() {
        const result = this.currentAnalysis;
        const info = result.disease_info;
        const confidence = Math.round(result.confidence * 100);
        
        return `PALM TREE DISEASE ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}

ANALYSIS RESULTS
================
Disease: ${info.name}
Confidence: ${confidence}%
Severity: ${info.severity}

DESCRIPTION
===========
${info.description}

SYMPTOMS
========
${info.symptoms.map(s => `• ${s}`).join('\n')}

RECOMMENDED TREATMENT
====================
${info.treatment.map(t => `• ${t}`).join('\n')}

PREVENTION MEASURES
==================
${info.prevention.map(p => `• ${p}`).join('\n')}

ALTERNATIVE POSSIBILITIES
========================
${result.alternatives.slice(1, 4).map(alt => 
    `• ${this.formatDiseaseName(alt.disease)}: ${Math.round(alt.confidence * 100)}%`
).join('\n')}

DISCLAIMER
==========
This analysis is provided for guidance only. For serious plant health concerns, 
please consult with a professional arborist or plant pathologist.

Report generated by Palm Tree Disease Detector AI System`;
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.currentTheme = newTheme;
        
        // Add smooth transition effect
        document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }

    async analyzeImage() {
        if (!this.selectedFile) return;

        this.showLoadingPanel();
        this.hideResults();
        this.hideError();

        // Show loading state on button
        const btnContent = this.analyzeBtn.querySelector('.btn-content');
        const btnLoader = this.analyzeBtn.querySelector('.btn-loader');
        btnContent.classList.add('hidden');
        btnLoader.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('image', this.selectedFile);

            // Enhanced progress simulation with realistic steps
            this.updateProgress(0, 'Preparing image...');
            await this.delay(800);
            
            this.updateProgress(25, 'Uploading to AI system...');
            await this.delay(600);
            
            this.updateProgress(50, 'Preprocessing with neural networks...');
            await this.delay(800);
            
            this.updateProgress(75, 'Analyzing disease patterns...');
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            this.updateProgress(95, 'Generating diagnosis...');
            await this.delay(400);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Analysis failed');
            }

            const result = await response.json();
            this.updateProgress(100, 'Analysis complete!');
            
            await this.delay(500);
            this.currentAnalysis = result;
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'An error occurred during analysis. Please try again.');
        } finally {
            this.hideLoadingPanel();
            // Reset button state
            btnContent.classList.remove('hidden');
            btnLoader.classList.add('hidden');
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PalmDiseaseDetector();
});
