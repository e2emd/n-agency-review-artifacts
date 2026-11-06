'use strict';
if (window.lucide) window.lucide.createIcons();
const modal = document.getElementById('figure-dialog');
const openButton = document.getElementById('open-figure');
if (typeof modal.showModal === 'function') {
  openButton.hidden = false;
  openButton.addEventListener('click', () => modal.showModal());
  document.getElementById('close-figure').addEventListener('click', () => modal.close());
  modal.addEventListener('click', (event) => {
    const box = modal.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) modal.close();
  });
}
