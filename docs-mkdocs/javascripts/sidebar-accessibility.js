document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarCollapse");

  if (!sidebar || !sidebarToggle) {
    return;
  }

  const syncSidebarState = () => {
    const desktopViewport = window.matchMedia("(min-width: 992px)").matches;
    const expanded = desktopViewport
      ? !sidebar.classList.contains("active")
      : sidebar.classList.contains("active");

    sidebarToggle.setAttribute("aria-expanded", String(expanded));
  };

  syncSidebarState();
  sidebarToggle.addEventListener("click", () => window.requestAnimationFrame(syncSidebarState));
  window.addEventListener("resize", syncSidebarState);
});
