document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("dropZone");
    const dropZoneContent = document.getElementById("dropZoneContent");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const btnClear = document.getElementById("btnClear");
    const detectBtn = document.getElementById("detectBtn");
    const resetBtn = document.getElementById("resetBtn");
    
    // Result panels
    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const errorAlert = document.getElementById("errorAlert");
    const errorTitle = document.getElementById("errorTitle");
    const errorMsg = document.getElementById("errorMsg");
    const resultContent = document.getElementById("resultContent");
    
    // Result details
    const badgeStatus = document.getElementById("badgeStatus");
    const resultTitle = document.getElementById("resultTitle");
    const predictionTime = document.getElementById("predictionTime");
    const confidenceValue = document.getElementById("confidenceValue");
    const progressBarFill = document.getElementById("progressBarFill");
    const descText = document.getElementById("descText");
    const symptomsList = document.getElementById("symptomsList");
    const causesList = document.getElementById("causesList");
    const treatmentText = document.getElementById("treatmentText");
    const preventionText = document.getElementById("preventionText");
    
    // History Sidebar
    const historySidebar = document.getElementById("historySidebar");
    const historyToggleBtn = document.getElementById("historyToggleBtn");
    const historyList = document.getElementById("historyList");
    const btnClearHistory = document.getElementById("btnClearHistory");

    let selectedFile = null;
    let currentImageBase64 = null;

    // Initialize History
    renderHistory();

    // ====================================
    // DRAG & DROP & FILE SELECTION
    // ====================================

    // Trigger file dialog
    dropZone.addEventListener("click", (e) => {
        // Prevent trigger if clicking on clear button inside preview
        if (e.target.closest("#btnClear")) return;
        if (selectedFile) return; // Ignore if image already uploaded
        fileInput.click();
    });

    fileInput.addEventListener("change", function () {
        handleFile(this.files[0]);
    });

    // Drag events
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!selectedFile) {
                dropZone.classList.add("dragover");
            }
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    // Drop handler
    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFile(file);
    });

    // Process selected/dropped file
    function handleFile(file) {
        if (!file) return;

        // Validation: Check type
        if (!file.type.startsWith("image/")) {
            showErrorUI("File Type Error", "Please upload a valid image file (PNG, JPG, JPEG, WEBP, BMP).");
            return;
        }

        // Validation: Check size (Max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showErrorUI("File Size Error", "The selected file is too large. Maximum size allowed is 10MB.");
            return;
        }

        selectedFile = file;
        hideErrorUI();

        const reader = new FileReader();
        reader.onload = function (e) {
            currentImageBase64 = e.target.result;
            previewImage.src = currentImageBase64;
            
            // Show preview & hide default content
            previewContainer.style.display = "block";
            dropZoneContent.style.opacity = "0";
            dropZoneContent.style.pointerEvents = "none";
            
            // Enable controls
            detectBtn.disabled = false;
            resetBtn.style.display = "inline-flex";
        };
        reader.readAsDataURL(file);
    }

    // Clear Uploaded Image
    btnClear.addEventListener("click", (e) => {
        e.stopPropagation();
        clearUpload();
    });

    function clearUpload() {
        selectedFile = null;
        currentImageBase64 = null;
        fileInput.value = "";
        
        // Hide preview & restore default content
        previewContainer.style.display = "none";
        previewImage.src = "";
        dropZoneContent.style.opacity = "1";
        dropZoneContent.style.pointerEvents = "auto";
        
        // Disable/hide controls
        detectBtn.disabled = true;
        resetBtn.style.display = "none";
        
        // Restore right side state
        showState("empty");
    }

    resetBtn.addEventListener("click", () => {
        clearUpload();
    });

    // ====================================
    // STATE TOGGLERS (RIGHT SIDE PANEL)
    // ====================================
    function showState(state) {
        emptyState.style.display = state === "empty" ? "flex" : "none";
        loadingState.style.display = state === "loading" ? "flex" : "none";
        errorAlert.style.display = state === "error" ? "flex" : "none";
        resultContent.style.display = state === "result" ? "block" : "none";
    }

    function showErrorUI(title, message) {
        errorTitle.innerText = title;
        errorMsg.innerText = message;
        showState("error");
    }

    function hideErrorUI() {
        if (errorAlert.style.display === "flex") {
            showState("empty");
        }
    }

    // ====================================
    // PREDICT DISEASE (API CALL)
    // ====================================
    window.showResult = async function() {
        if (!selectedFile) {
            showErrorUI("Upload Required", "Please select or drop an image first.");
            return;
        }

        showState("loading");
        detectBtn.disabled = true;

        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const response = await fetch("https://crop-disease-detector-backend-njdi.onrender.com/predict", {
            method: "POST",
            body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();

            // Handle invalid case (e.g. not a leaf)
            if (data.valid === false) {
                showErrorUI("Invalid Leaf Image", data.error);
                detectBtn.disabled = false;
                return;
            }

            // Successfully analyzed
            displayResult(data);
            
            // Save to LocalStorage History Log
            saveToHistory(data.details.name, data.confidence, data, currentImageBase64);

            detectBtn.disabled = false;

        } catch (error) {
            console.error("Backend request failed: ", error);
            showErrorUI(
            "Connection Failed",
            "Unable to connect to the Crop Disease Detection server."
            );
            detectBtn.disabled = false;
        }
    };

    // Update Result Panel DOM
    function displayResult(data) {
        showState("result");

        // Title and badge
        resultTitle.innerText = data.details.name;
        predictionTime.innerText = data.prediction_time;
        
        // Badge type
        badgeStatus.innerText = data.details.status;
        badgeStatus.className = `badge-status ${data.details.status}`;

        // Confidence progress bar
        confidenceValue.innerText = `${data.confidence}%`;
        // Animate fill bar
        progressBarFill.style.width = "0%";
        setTimeout(() => {
            progressBarFill.style.width = `${data.confidence}%`;
        }, 100);

        // Description
        descText.innerText = data.details.description;

        // Symptoms list
        symptomsList.innerHTML = "";
        data.details.symptoms.forEach(symptom => {
            const li = document.createElement("li");
            li.innerText = symptom;
            symptomsList.appendChild(li);
        });

        // Causes list
        causesList.innerHTML = "";
        data.details.causes.forEach(cause => {
            const li = document.createElement("li");
            li.innerText = cause;
            causesList.appendChild(li);
        });

        // Treatment
        treatmentText.innerText = data.details.treatment;

        // Prevention
        preventionText.innerText = data.details.prevention;
    }

    // ====================================
    // PREDICTION HISTORY LOGGER
    // ====================================

    // Toggle Sidebar
    historyToggleBtn.addEventListener("click", () => {
        historySidebar.classList.toggle("active");
    });

    // Close sidebar when clicking outside on dashboard
    document.addEventListener("click", (e) => {
        if (!historySidebar.contains(e.target) && !historyToggleBtn.contains(e.target)) {
            historySidebar.classList.remove("active");
        }
    });

    // Save item
    function saveToHistory(title, confidence, resultDetails, imageBase64) {
        let history = JSON.parse(localStorage.getItem("crop_history") || "[]");
        
        const newItem = {
            id: Date.now().toString(),
            title: title,
            confidence: confidence,
            details: resultDetails,
            image: imageBase64,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + " - " + new Date().toLocaleDateString()
        };

        // Prepend new item
        history.unshift(newItem);

        // Limit local history size to 8 items to prevent exceeding LocalStorage limits
        if (history.length > 8) {
            history.pop();
        }

        try {
            localStorage.setItem("crop_history", JSON.stringify(history));
        } catch (e) {
            console.warn("LocalStorage quota warning. Truncating history further.");
            if (history.length > 3) {
                history = history.slice(0, 3);
                localStorage.setItem("crop_history", JSON.stringify(history));
            }
        }

        renderHistory();
        // Automatically show history slide-in brief review
        historySidebar.classList.add("active");
    }

    // Render list
    function renderHistory() {
        const history = JSON.parse(localStorage.getItem("crop_history") || "[]");
        historyList.innerHTML = "";

        if (history.length === 0) {
            historyList.innerHTML = '<p class="no-history-text">No past predictions.</p>';
            return;
        }

        history.forEach(item => {
            const div = document.createElement("div");
            div.className = "history-item";
            div.dataset.id = item.id;
            
            div.innerHTML = `
                <div class="history-item-top">
                    <span class="history-item-title">${item.title}</span>
                    <span class="history-item-conf">${item.confidence}%</span>
                </div>
                <div class="history-item-time">${item.timestamp}</div>
            `;

            div.addEventListener("click", () => {
                loadHistoryItem(item);
            });

            historyList.appendChild(div);
        });
    }

    // Load old item back to UI
    function loadHistoryItem(item) {
        selectedFile = dataURLtoFile(item.image, "history_img.jpg");
        currentImageBase64 = item.image;
        
        previewImage.src = currentImageBase64;
        previewContainer.style.display = "block";
        dropZoneContent.style.opacity = "0";
        dropZoneContent.style.pointerEvents = "none";
        
        detectBtn.disabled = false;
        resetBtn.style.display = "inline-flex";

        displayResult(item.details);
        historySidebar.classList.remove("active");
    }

    // Helper: convert DataURL to File object for subsequent diagnostic requests if re-run
    function dataURLtoFile(dataurl, filename) {
        let arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
        while(n--){
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new File([u8arr], filename, {type:mime});
    }

    // Clear history
    btnClearHistory.addEventListener("click", () => {
        if (confirm("Are you sure you want to clear your prediction history?")) {
            localStorage.removeItem("crop_history");
            renderHistory();
        }
    });
});