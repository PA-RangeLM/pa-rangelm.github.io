const gif = document.querySelector('#qualitative-gif');
const replay = document.querySelector('#replay-gif');

if (gif && replay) {
  replay.addEventListener('click', () => {
    const source = gif.dataset.src;
    replay.disabled = true;
    gif.src = `${source}?replay=${Date.now()}`;
    window.setTimeout(() => { replay.disabled = false; }, 700);
  });
}

const navLinks = [...document.querySelectorAll('.site-nav a')];
const sections = navLinks.map((link) => document.querySelector(link.getAttribute('href'))).filter(Boolean);

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navLinks.forEach((link) => link.classList.toggle('is-active', link.getAttribute('href') === `#${entry.target.id}`));
    });
  }, { rootMargin: '-25% 0px -65% 0px' });
  sections.forEach((section) => observer.observe(section));
}

const checkpointParts = [
  'assets/downloads/pa-rangelm-pcn-10p20.part-00',
  'assets/downloads/pa-rangelm-pcn-10p20.part-01',
  'assets/downloads/pa-rangelm-pcn-10p20.part-02',
  'assets/downloads/pa-rangelm-pcn-10p20.part-03'
];
const checkpointSize = 155741220;
let checkpointDownloadActive = false;

async function fetchPart(url, onChunk) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed (${response.status})`);
  if (!response.body) {
    const fallback = await response.arrayBuffer();
    onChunk(fallback.byteLength);
    return new Uint8Array(fallback);
  }
  const reader = response.body.getReader();
  const chunks = [];
  let length = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    length += value.byteLength;
    onChunk(value.byteLength);
  }
  const merged = new Uint8Array(length);
  let offset = 0;
  chunks.forEach((chunk) => { merged.set(chunk, offset); offset += chunk.byteLength; });
  return merged;
}

async function downloadCheckpoint() {
  if (checkpointDownloadActive) return;
  checkpointDownloadActive = true;
  const panel = document.querySelector('.checkpoint-status');
  const progress = document.querySelector('#checkpoint-progress');
  const percent = document.querySelector('#checkpoint-percent');
  const message = document.querySelector('#checkpoint-message');
  panel.hidden = false;
  panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
  let received = 0;
  const update = (bytes) => {
    received += bytes;
    progress.value = received;
    percent.textContent = `${Math.min(100, Math.round((received / checkpointSize) * 100))}%`;
  };
  try {
    const buffers = [];
    for (let i = 0; i < checkpointParts.length; i += 1) {
      message.textContent = `Downloading weight part ${i + 1} of ${checkpointParts.length}…`;
      buffers.push(await fetchPart(checkpointParts[i], update));
    }
    message.textContent = 'Combining anonymous checkpoint locally…';
    const blob = new Blob(buffers, { type: 'application/octet-stream' });
    const link = document.createElement('a');
    const objectUrl = URL.createObjectURL(blob);
    link.href = objectUrl;
    link.download = 'pa_rangelm_pcn_rotated_10p20.pth';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 10000);
    message.textContent = 'Checkpoint ready — SHA-256 starts with b72f5191.';
    percent.textContent = '100%';
  } catch (error) {
    message.textContent = `Checkpoint download failed: ${error.message}`;
    percent.textContent = 'Retry';
  } finally {
    checkpointDownloadActive = false;
  }
}

document.querySelectorAll('.checkpoint-trigger').forEach((button) => button.addEventListener('click', downloadCheckpoint));
