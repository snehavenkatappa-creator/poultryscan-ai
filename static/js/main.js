/**
 * PoultryScan AI - Main Client JavaScript
 * Handles: Mobile Drawer, Drag-and-Drop Upload, Client Previews,
 * Scanning Indicators, and Counter Micro-Animations.
 */

document.addEventListener("DOMContentLoaded", () => {
  initMobileNav();
  initUploadDropzone();
  initAnimatedCounters();
});

/* --------------------------------------------------------------------------
   Mobile Navigation Drawer
   -------------------------------------------------------------------------- */
function initMobileNav() {
  const menuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("appSidebar");
  const overlay = document.getElementById("sidebarOverlay");

  if (!menuBtn || !sidebar || !overlay) return;

  function toggleSidebar() {
    sidebar.classList.toggle("open");
    overlay.classList.toggle("active");
  }

  function closeSidebar() {
    sidebar.classList.remove("open");
    overlay.classList.remove("active");
  }

  menuBtn.addEventListener("click", toggleSidebar);
  overlay.addEventListener("click", closeSidebar);
}

/* --------------------------------------------------------------------------
   Drag and Drop Upload & Scanning State
   -------------------------------------------------------------------------- */
function initUploadDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const cameraInput = document.getElementById("cameraInput");
  const dropPrompt = document.getElementById("dropPrompt");
  const previewWrapper = document.getElementById("previewWrapper");
  const imagePreview = document.getElementById("imagePreview");
  const previewMeta = document.getElementById("previewMeta");
  const removeBtn = document.getElementById("removeBtn");
  const submitBtn = document.getElementById("submitBtn");
  const uploadForm = document.getElementById("uploadForm");
  const uploadCard = document.getElementById("uploadCard");
  const scanningCard = document.getElementById("scanningCard");
  const scannerStatus = document.getElementById("scannerStatus");

  if (!dropzone || !fileInput || !uploadForm) return;

  // Open file dialog on dropzone click
  dropzone.addEventListener("click", (e) => {
    if (e.target !== removeBtn) {
      fileInput.click();
    }
  });

  // Handle Drag & Drop Events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      handleSelectedFile(files[0]);
    }
  });

  // File Input Change
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleSelectedFile(fileInput.files[0]);
    }
  });

  // Camera Input Change
  if (cameraInput) {
    cameraInput.addEventListener("change", () => {
      if (cameraInput.files.length > 0) {
        fileInput.files = cameraInput.files;
        handleSelectedFile(cameraInput.files[0]);
      }
    });
  }

  // Remove Selected Image
  if (removeBtn) {
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.value = "";
      if (cameraInput) cameraInput.value = "";
      previewWrapper.style.display = "none";
      dropPrompt.style.display = "flex";
      removeBtn.style.display = "none";
      submitBtn.disabled = true;
    });
  }

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (PNG, JPG, JPEG, WEBP).");
      return;
    }

    if (file.size > 8 * 1024 * 1024) {
      alert("File size exceeds 8 MB. Please choose a smaller photo.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewWrapper.style.display = "inline-block";
      dropPrompt.style.display = "none";
      if (removeBtn) removeBtn.style.display = "inline-flex";
      submitBtn.disabled = false;

      const sizeKb = (file.size / 1024).toFixed(1);
      previewMeta.textContent = `${file.name} · ${sizeKb} KB`;
    };
    reader.readAsDataURL(file);
  }

  // Form Submit -> Radar Scanning Animation
  uploadForm.addEventListener("submit", () => {
    submitBtn.disabled = true;
    uploadCard.style.display = "none";
    if (scanningCard) {
      scanningCard.style.display = "block";
      startScannerStatusCycles();
    }
  });

  function startScannerStatusCycles() {
    const statuses = [
      "Preprocessing image to 224x224 RGB tensor...",
      "Extracting spatial feature maps via MobileNetV2 backbone...",
      "Running Global Average Pooling & Dropout regularization...",
      "Calculating 4-Class Softmax diagnostic probabilities...",
      "Archiving scan record to SQLite audit log...",
    ];
    let i = 0;
    setInterval(() => {
      i = (i + 1) % statuses.length;
      if (scannerStatus) {
        scannerStatus.textContent = statuses[i];
      }
    }, 1100);
  }
}

/* --------------------------------------------------------------------------
   Animated Number Counters for Dashboard & Home KPIs
   -------------------------------------------------------------------------- */
function initAnimatedCounters() {
  const counters = document.querySelectorAll("[data-count]");
  counters.forEach((counter) => {
    const target = parseFloat(counter.getAttribute("data-count")) || 0;
    if (target === 0) return;

    let current = 0;
    const step = target / 30;
    const interval = setInterval(() => {
      current += step;
      if (current >= target) {
        current = target;
        clearInterval(interval);
      }
      counter.textContent = Number.isInteger(target)
        ? Math.floor(current)
        : current.toFixed(1);
    }, 30);
  });
}
