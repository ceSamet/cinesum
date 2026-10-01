// CineSum AI — Advanced Cinematic Frontend Logic with Dynamic Duration Slider Adaptation

let selectedCategory = 'action';
let selectedVideoAlias = 'video1';
let progressPollInterval = null;
let aiAnalyticsChartInstance = null;
let currentSummaryData = null;
let currentViewMode = 'full'; // 'full' or 'summary'

document.addEventListener('DOMContentLoaded', () => {
    fetchAvailableVideos();
    setupLiveHudListener();
    setupVideoSelectListener();
});

// Generate unique task ID
function generateTaskId() {
    return 'task_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now();
}

// Listen for video selection dropdown changes to load full video info instantly
function setupVideoSelectListener() {
    const select = document.getElementById('videoSelect');
    select.addEventListener('change', (e) => {
        selectedVideoAlias = e.target.value;
        if (selectedVideoAlias) {
            loadVideoInfo(selectedVideoAlias);
        }
    });
}

// Instantly load raw full video info (scores, timeline, subtitles) upon selection
async function loadVideoInfo(videoAlias) {
    try {
        const res = await fetch(`/api/video-info/${videoAlias}`);
        const data = await res.json();

        if (data.success && data.all_scenes && data.all_scenes.length > 0) {
            currentSummaryData = {
                video_alias: videoAlias,
                output_video_url: data.video_url,
                all_scenes: data.all_scenes,
                selected_scenes: [], // No summary selected yet
            };

            // Dynamically adjust duration slider MAX to the total duration of the selected video
            updateDurationSliderLimits(data.all_scenes);

            currentViewMode = 'full';
            renderCurrentViewMode();
        } else {
            console.warn(`[VİDEO BİLGİSİ] ${videoAlias} henüz analiz edilmemiş.`);
        }
    } catch (err) {
        console.warn('Video ön bilgisi henüz yüklenemedi:', err);
    }
}

// Dynamically set Duration Slider MAX to total video length in seconds
function updateDurationSliderLimits(allScenes) {
    const slider = document.getElementById('durationSlider');
    const display = document.getElementById('durValueDisplay');

    if (!allScenes || allScenes.length === 0) return;

    const totalDurSec = Math.round(allScenes.reduce((acc, sc) => acc + sc.duration_seconds, 0));

    if (totalDurSec > 0) {
        slider.min = "5";
        slider.max = totalDurSec.toString();
        
        // If current slider value exceeds total video duration, clamp it
        if (parseInt(slider.value) > totalDurSec) {
            slider.value = totalDurSec.toString();
        }

        const mins = Math.floor(totalDurSec / 60);
        const secs = totalDurSec % 60;
        const formattedTotal = mins > 0 ? `${mins}dk ${secs}s` : `${secs}s`;

        display.textContent = `${slider.value} saniye (Maks: ${formattedTotal})`;
    }
}

// Fetch list of processed videos from backend
async function fetchAvailableVideos() {
    try {
        const res = await fetch('/api/videos');
        const data = await res.json();
        const select = document.getElementById('videoSelect');
        select.innerHTML = '';

        if (data.videos && data.videos.length > 0) {
            data.videos.forEach((v, idx) => {
                const opt = document.createElement('option');
                opt.value = v;
                opt.textContent = `${v} (${idx + 1}/${data.videos.length})`;
                select.appendChild(opt);
            });
            selectedVideoAlias = data.videos[0];
            await loadVideoInfo(selectedVideoAlias);
        } else {
            select.innerHTML = '<option value="">Hiç video bulunamadı</option>';
        }
    } catch (err) {
        console.error('Video listesi alınamadı:', err);
    }
}

// Tab Switching (Select vs Upload)
function switchTab(tabName) {
    const selectBtn = document.getElementById('tabSelectBtn');
    const uploadBtn = document.getElementById('tabUploadBtn');
    const selectContent = document.getElementById('tabSelectContent');
    const uploadContent = document.getElementById('tabUploadContent');

    if (tabName === 'select') {
        selectBtn.classList.add('active');
        uploadBtn.classList.remove('active');
        selectContent.classList.remove('hidden');
        uploadContent.classList.add('hidden');
    } else {
        uploadBtn.classList.add('active');
        selectBtn.classList.remove('active');
        uploadContent.classList.remove('hidden');
        selectContent.classList.add('hidden');
    }
}

// Category Selection
function selectCategory(cat) {
    selectedCategory = cat;
    ['Action', 'Dialogue', 'Importance', 'Custom'].forEach(c => {
        const card = document.getElementById(`cat${c}`);
        if (card) card.classList.remove('active');
    });

    const activeCard = document.getElementById(`cat${cat.charAt(0).toUpperCase() + cat.slice(1)}`);
    if (activeCard) activeCard.classList.add('active');

    const promptGroup = document.getElementById('promptBoxGroup');
    if (cat === 'custom') {
        promptGroup.classList.remove('hidden');
    } else {
        promptGroup.classList.add('hidden');
    }
}

// Duration slider & checkbox controls
function updateDurationValue(val) {
    const slider = document.getElementById('durationSlider');
    const maxVal = slider.max;
    
    const maxMins = Math.floor(maxVal / 60);
    const maxSecs = maxVal % 60;
    const formattedTotal = maxMins > 0 ? `${maxMins}dk ${maxSecs}s` : `${maxSecs}s`;

    document.getElementById('durValueDisplay').textContent = `${val} saniye (Maks: ${formattedTotal})`;
}

function toggleSelectAll(isChecked) {
    const sliderGroup = document.getElementById('sliderGroup');
    if (isChecked) {
        sliderGroup.style.opacity = '0.4';
        sliderGroup.style.pointerEvents = 'none';
    } else {
        sliderGroup.style.opacity = '1';
        sliderGroup.style.pointerEvents = 'auto';
    }
}

// Playback Speed Controller
function setPlaybackSpeed(speed, btnElement) {
    const player = document.getElementById('summaryVideoPlayer');
    if (player) {
        player.playbackRate = speed;
    }
    const btns = document.querySelectorAll('.speed-btn');
    btns.forEach(b => b.classList.remove('active'));
    if (btnElement) btnElement.classList.add('active');
}

// Theater Mode Toggle
function toggleTheaterMode(forceState) {
    const body = document.body;
    const backdrop = document.getElementById('theaterBackdrop');
    
    if (typeof forceState === 'boolean') {
        if (forceState) {
            body.classList.add('theater-mode-active');
            backdrop.classList.remove('hidden');
        } else {
            body.classList.remove('theater-mode-active');
            backdrop.classList.add('hidden');
        }
    } else {
        const isActive = body.classList.toggle('theater-mode-active');
        if (isActive) {
            backdrop.classList.remove('hidden');
        } else {
            backdrop.classList.add('hidden');
        }
    }
}

// Dual View Mode Switcher (Full Original Video vs Summary Video)
function switchPlayerViewMode(mode) {
    currentViewMode = mode;

    const fullBtn = document.getElementById('viewModeFullBtn');
    const summaryBtn = document.getElementById('viewModeSummaryBtn');

    if (mode === 'full') {
        fullBtn.classList.add('active');
        summaryBtn.classList.remove('active');
    } else {
        summaryBtn.classList.add('active');
        fullBtn.classList.remove('active');
    }

    if (currentSummaryData) {
        renderCurrentViewMode();
    }
}

// Real-time Progress Polling Function
function startProgressPolling(taskId, modalTitleText = "Yapay Zekâ Analiz Ediyor") {
    const modal = document.getElementById('processingModal');
    const titleEl = document.getElementById('modalTitle');
    const stageDescEl = document.getElementById('modalStageDesc');
    const fillEl = document.getElementById('progressBarFill');
    const percentEl = document.getElementById('progressPercent');
    const detailEl = document.getElementById('progressDetail');

    titleEl.textContent = modalTitleText;
    fillEl.style.width = '0%';
    percentEl.textContent = '0%';
    stageDescEl.textContent = 'İşlem başlatılıyor...';
    detailEl.textContent = 'Hazırlanıyor...';

    ['stepScene', 'stepClip', 'stepAudio', 'stepExport'].forEach(s => {
        const el = document.getElementById(s);
        if (el) el.className = 'stage-step-item';
    });

    modal.classList.remove('hidden');

    if (progressPollInterval) clearInterval(progressPollInterval);

    progressPollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/progress/${taskId}`);
            if (res.ok) {
                const progData = await res.json();

                const p = progData.progress || 0;
                fillEl.style.width = `${p}%`;
                percentEl.textContent = `${p}%`;
                stageDescEl.textContent = progData.stage_desc || 'İşleniyor...';
                detailEl.textContent = progData.detail || '';

                const activeStep = progData.active_step;
                const stepsOrder = ['stepScene', 'stepClip', 'stepAudio', 'stepExport'];
                let activeIdx = stepsOrder.indexOf(activeStep);

                stepsOrder.forEach((stepId, idx) => {
                    const stepEl = document.getElementById(stepId);
                    if (stepEl) {
                        if (idx < activeIdx) {
                            stepEl.className = 'stage-step-item completed';
                        } else if (idx === activeIdx) {
                            stepEl.className = 'stage-step-item active';
                        } else {
                            stepEl.className = 'stage-step-item';
                        }
                    }
                });
            }
        } catch (e) {
            console.error('Progress polling hatası:', e);
        }
    }, 300);
}

function stopProgressPolling() {
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
        progressPollInterval = null;
    }
    const modal = document.getElementById('processingModal');
    modal.classList.add('hidden');
}

// Handle Drag & Drop Upload
async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0];

    const taskId = generateTaskId();
    startProgressPolling(taskId, "Yeni Video Yükleniyor ve Analiz Ediliyor");

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`/api/upload?task_id=${taskId}`, {
            method: 'POST',
            body: formData,
        });

        const data = await res.json();
        stopProgressPolling();

        if (res.ok && data.success) {
            alert(`🎉 Video Başarıyla Yüklendi ve Analiz Edildi: ${data.video_alias}`);
            await fetchAvailableVideos();
            switchTab('select');
            document.getElementById('videoSelect').value = data.video_alias;
            selectedVideoAlias = data.video_alias;
            await loadVideoInfo(data.video_alias);
        } else {
            alert(`[HATA] Video yüklenemedi: ${data.detail || data.message || 'Bilinmeyen hata'}`);
        }
    } catch (err) {
        stopProgressPolling();
        alert(`[HATA] Yükleme sırasında ağ hatası: ${err}`);
    }
}

// Main Summarization Action Trigger
async function triggerSummarization() {
    const taskId = generateTaskId();
    startProgressPolling(taskId, "Özet Video Üretiliyor");

    const isSelectAll = document.getElementById('selectAllCheckbox').checked;
    const targetDur = isSelectAll ? 0 : parseFloat(document.getElementById('durationSlider').value);
    const customPrompt = document.getElementById('customPromptInput').value.trim();
    const narrativeMode = document.getElementById('narrativeAiCheckbox').checked ? 'rag_llm' : 'local';

    const payload = {
        task_id: taskId,
        video_alias: selectedVideoAlias,
        category: selectedCategory,
        custom_prompt: customPrompt,
        target_duration_sec: targetDur,
        narrative_mode: narrativeMode,
    };

    try {
        const res = await fetch('/api/summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await res.json();
        stopProgressPolling();

        if (res.ok && data.success) {
            currentSummaryData = data;
            renderSummaryResult(data);
            switchPlayerViewMode('summary'); // Automatically switch to summary view mode on summary generation!
        } else {
            alert(`[HATA] Özet oluşturulamadı: ${data.detail || data.message || 'Bilinmeyen hata'}`);
        }
    } catch (err) {
        stopProgressPolling();
        alert(`[HATA] Sunucuyla iletişim hatası: ${err}`);
    }
}

// Seek Video Player to timestamp in seconds
function seekToTimestamp(seconds) {
    const player = document.getElementById('summaryVideoPlayer');
    if (player) {
        player.currentTime = seconds;
        player.play();
    }
}

// Live Floating HUD Video Overlay Listener (Score Badges Only)
function setupLiveHudListener() {
    const player = document.getElementById('summaryVideoPlayer');
    const hudOverlay = document.getElementById('liveHudOverlay');
    const hudShotBadge = document.getElementById('hudShotBadge');
    const hudActionScore = document.getElementById('hudActionScore');
    const hudDialogueScore = document.getElementById('hudDialogueScore');
    const hudImportanceScore = document.getElementById('hudImportanceScore');

    player.addEventListener('timeupdate', () => {
        if (!currentSummaryData) return;

        const currentTime = player.currentTime;
        const selectedScenes = currentSummaryData.selected_scenes || [];
        const allScenes = (currentSummaryData.all_scenes && currentSummaryData.all_scenes.length > 0)
            ? currentSummaryData.all_scenes
            : selectedScenes;

        let currentShot = null;

        if (currentViewMode === 'summary') {
            let runningTime = 0.0;
            for (let sc of selectedScenes) {
                const scDur = sc.duration_seconds;
                if (currentTime >= runningTime && currentTime <= (runningTime + scDur)) {
                    currentShot = sc;
                    break;
                }
                runningTime += scDur;
            }
        } else {
            currentShot = allScenes.find(sc => currentTime >= sc.start_seconds && currentTime <= sc.end_seconds);
        }

        if (!currentShot && allScenes.length > 0) {
            currentShot = allScenes[0];
        }

        if (currentShot) {
            hudShotBadge.textContent = `Shot ${currentShot.scene_id}`;

            const actionVal = (currentShot.action_score || currentShot.query_max_similarity || 0.1).toFixed(2);
            const dialogueVal = (currentShot.dialogue_score || (currentShot.speech_ratio ? currentShot.speech_ratio * 0.8 : 0.05)).toFixed(2);
            const importanceVal = (currentShot.importance_score || 0.2).toFixed(2);

            hudActionScore.textContent = `💥 Aksiyon: ${actionVal}`;
            hudDialogueScore.textContent = `💬 Diyalog: ${dialogueVal}`;
            hudImportanceScore.textContent = `⭐ Önem: ${importanceVal}`;

            hudOverlay.classList.remove('hidden');
        }
    });
}

// Render Results on UI
function renderSummaryResult(data) {
    const downloadBtn = document.getElementById('downloadBtn');
    const narrativeBadge = document.getElementById('narrativeStatusBadge');
    downloadBtn.classList.remove('hidden');
    downloadBtn.href = data.output_video_url;

    const llmStatus = data.llm_selection?.status || 'disabled';
    narrativeBadge.className = `narrative-status status-${llmStatus}`;
    if (llmStatus === 'applied') {
        narrativeBadge.textContent = 'AI anlatı uygulandı';
    } else if (llmStatus === 'cached') {
        narrativeBadge.textContent = 'AI anlatı cache';
    } else if (llmStatus === 'fallback_local') {
        narrativeBadge.textContent = 'Yerel güvenli sonuç';
    } else {
        narrativeBadge.textContent = 'Yerel mod';
    }

    renderCurrentViewMode();
}

// Render UI based on currentViewMode ('full' vs 'summary')
function renderCurrentViewMode() {
    if (!currentSummaryData) return;

    const data = currentSummaryData;
    const placeholder = document.getElementById('playerPlaceholder');
    const player = document.getElementById('summaryVideoPlayer');
    const scenesSection = document.getElementById('scenesSection');
    const scenesList = document.getElementById('scenesList');
    const scrubberTitle = document.getElementById('scrubberTitleText');
    const chartTitle = document.getElementById('chartTitleText');

    placeholder.classList.add('hidden');
    player.classList.remove('hidden');

    const selectedScenes = data.selected_scenes || [];
    const allScenes = (data.all_scenes && data.all_scenes.length > 0) ? data.all_scenes : selectedScenes;

    if (currentViewMode === 'full') {
        // MODE 1: FULL UNTRIMMED ORIGINAL VIDEO
        player.src = `/dataset/video/${data.video_alias}.mp4`;
        scrubberTitle.innerHTML = `<i class="fa-solid fa-sliders"></i> Renk Kodlu Çekim Çizelgesi — 🎬 Orijinal Tam Video (${allScenes.length} Çekim)`;
        chartTitle.innerHTML = `<i class="fa-solid fa-chart-line"></i> Canlı Yapay Zekâ Skor Dağılım Dalga Grafiği — 🎬 Orijinal Tam Video (${allScenes.length} Çekim)`;

        renderScrubberTimelineBar(allScenes, selectedScenes);
        renderAnalyticsChart(allScenes, "Orijinal Tam Video");
        renderTranscriptViewer(allScenes);

    } else {
        // MODE 2: GENERATED SUMMARY VIDEO ONLY
        player.src = data.output_video_url;
        scrubberTitle.innerHTML = `<i class="fa-solid fa-sliders"></i> Renk Kodlu Çekim Çizelgesi — ⚡ Yapay Zekâ Özeti (${selectedScenes.length} Kesilmiş Çekim)`;
        chartTitle.innerHTML = `<i class="fa-solid fa-chart-line"></i> Canlı Yapay Zekâ Skor Dağılım Dalga Grafiği — ⚡ Yapay Zekâ Özeti (${selectedScenes.length} Kesilmiş Çekim)`;

        renderScrubberTimelineBar(selectedScenes, selectedScenes);
        renderAnalyticsChart(selectedScenes, "Kesilmiş Yapay Zekâ Özeti");
        renderTranscriptViewer(selectedScenes);
    }

    // Render Timeline Cards for Selected Summary Shots
    scenesList.innerHTML = '';
    if (selectedScenes.length > 0) {
        selectedScenes.forEach((sc) => {
            const card = document.createElement('div');
            card.className = 'scene-card';

            const scoreVal = sc.query_max_similarity || sc.action_score || sc.importance_score || sc.dialogue_score || 0;

            card.innerHTML = `
                <div class="scene-info">
                    <h5>Shot ${sc.scene_id} (${sc.duration_seconds.toFixed(1)}s)</h5>
                    <p><i class="fa-regular fa-clock"></i> ${sc.start_timecode} &rarr; ${sc.end_timecode}</p>
                    ${sc.transcript_text ? `<p><i class="fa-solid fa-quote-left"></i> "${sc.transcript_text.substring(0, 60)}..."</p>` : ''}
                </div>
                <div class="scene-score">
                    ⚡ ${scoreVal.toFixed(3)}
                </div>
            `;
            card.addEventListener('click', () => {
                if (currentViewMode === 'full') {
                    seekToTimestamp(sc.start_seconds);
                } else {
                    let runningTime = 0.0;
                    for (let item of selectedScenes) {
                        if (item.scene_id === sc.scene_id) break;
                        runningTime += item.duration_seconds;
                    }
                    seekToTimestamp(runningTime);
                }
            });
            scenesList.appendChild(card);
        });
        scenesSection.classList.remove('hidden');
    }
}

// 1. Color-Coded Interactive Scrubber Bar Renderer
function renderScrubberTimelineBar(displayScenes, selectedScenes = []) {
    const scrubberContainer = document.getElementById('scrubberContainer');
    const scrubberBar = document.getElementById('scrubberBar');

    if (!displayScenes || displayScenes.length === 0) {
        scrubberContainer.classList.add('hidden');
        return;
    }

    scrubberBar.innerHTML = '';
    const totalDuration = displayScenes.reduce((acc, sc) => acc + sc.duration_seconds, 0) || 1.0;
    const selectedIds = new Set(selectedScenes.map(s => s.scene_id));

    let accumulatedPct = 0;
    let accumulatedTime = 0.0;

    displayScenes.forEach((sc) => {
        const segPct = (sc.duration_seconds / totalDuration) * 100;
        const seg = document.createElement('div');
        const isSelected = selectedIds.has(sc.scene_id);

        const actionScore = sc.action_score || sc.query_max_similarity || 0;
        const dialogueScore = sc.dialogue_score || (sc.speech_ratio ? sc.speech_ratio * 0.8 : 0);

        let colorClass = 'segment-neutral';

        if (actionScore > 0.45 && dialogueScore > 0.35) {
            colorClass = 'segment-overlap';
        } else if (actionScore >= dialogueScore && actionScore > 0.30) {
            colorClass = 'segment-action';
        } else if (dialogueScore > actionScore && dialogueScore > 0.30) {
            colorClass = 'segment-dialogue';
        }

        seg.className = `scrubber-segment ${colorClass}`;
        if (!isSelected && selectedScenes.length > 0 && currentViewMode === 'full') {
            seg.style.opacity = '0.35';
        }

        seg.style.left = `${accumulatedPct}%`;
        seg.style.width = `${Math.max(0.4, segPct)}%`;
        seg.title = `Shot ${sc.scene_id} (${sc.start_timecode} -> ${sc.end_timecode}) ${isSelected ? '[ÖZETİN PARÇASI]' : ''}`;

        const seekTargetSec = (currentViewMode === 'full') ? sc.start_seconds : accumulatedTime;

        seg.addEventListener('click', (e) => {
            e.stopPropagation();
            seekToTimestamp(seekTargetSec);
        });

        scrubberBar.appendChild(seg);
        accumulatedPct += segPct;
        accumulatedTime += sc.duration_seconds;
    });

    scrubberContainer.classList.remove('hidden');
}

// 2. AI Analytics Wave Chart (Chart.js Line Graph)
function renderAnalyticsChart(displayScenes, chartLabelSuffix = "") {
    const chartSection = document.getElementById('chartSection');
    const ctx = document.getElementById('aiAnalyticsChart').getContext('2d');

    if (!displayScenes || displayScenes.length === 0) {
        chartSection.classList.add('hidden');
        return;
    }

    const labels = displayScenes.map(sc => `Shot ${sc.scene_id}`);
    const actionData = displayScenes.map(sc => (sc.action_score || sc.query_max_similarity || 0.1).toFixed(2));
    const dialogueData = displayScenes.map(sc => (sc.dialogue_score || (sc.speech_ratio ? sc.speech_ratio * 0.8 : 0.05)).toFixed(2));
    const importanceData = displayScenes.map(sc => (sc.importance_score || 0.2).toFixed(2));

    if (aiAnalyticsChartInstance) {
        aiAnalyticsChartInstance.destroy();
    }

    aiAnalyticsChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: `Aksiyon Skoru (${chartLabelSuffix})`,
                    data: actionData,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2,
                },
                {
                    label: `Diyalog Skoru (${chartLabelSuffix})`,
                    data: dialogueData,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2,
                },
                {
                    label: `Önem Skoru (${chartLabelSuffix})`,
                    data: importanceData,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.10)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', font: { size: 9 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 20 },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: {
                    min: 0,
                    max: 1.0,
                    ticks: { color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            },
            onClick: (e, elements) => {
                if (elements && elements.length > 0) {
                    const idx = elements[0].index;
                    if (displayScenes[idx]) {
                        if (currentViewMode === 'full') {
                            seekToTimestamp(displayScenes[idx].start_seconds);
                        } else {
                            let runningTime = 0.0;
                            for (let i = 0; i < idx; i++) {
                                runningTime += displayScenes[i].duration_seconds;
                            }
                            seekToTimestamp(runningTime);
                        }
                    }
                }
            }
        }
    });

    chartSection.classList.remove('hidden');
}

// 3. Searchable Transcript Viewer
function renderTranscriptViewer(displayScenes) {
    const transcriptSection = document.getElementById('transcriptSection');
    const transcriptList = document.getElementById('transcriptList');

    if (!displayScenes || displayScenes.length === 0) {
        transcriptSection.classList.add('hidden');
        return;
    }

    transcriptList.innerHTML = '';
    let count = 0;
    let accumulatedTime = 0.0;

    displayScenes.forEach(sc => {
        if (sc.transcript_text && sc.transcript_text.trim().length > 0) {
            count++;
            const row = document.createElement('div');
            row.className = 'transcript-row';
            row.dataset.text = sc.transcript_text.toLowerCase();

            row.innerHTML = `
                <span class="time-tag">${sc.start_timecode}</span>
                <span class="transcript-text-content">"${sc.transcript_text}"</span>
            `;

            const seekSec = (currentViewMode === 'full') ? sc.start_seconds : accumulatedTime;
            row.addEventListener('click', () => seekToTimestamp(seekSec));
            transcriptList.appendChild(row);
        }
        accumulatedTime += sc.duration_seconds;
    });

    if (count > 0) {
        transcriptSection.classList.remove('hidden');
    } else {
        transcriptSection.classList.add('hidden');
    }
}

// Filter Transcripts live on user typing
function filterTranscripts(query) {
    const rows = document.querySelectorAll('.transcript-row');
    const q = query.toLowerCase().trim();

    rows.forEach(r => {
        const text = r.dataset.text || '';
        if (q === '' || text.includes(q)) {
            r.style.display = 'flex';
        } else {
            r.style.display = 'none';
        }
    });
}
