// FastAgentic documentation custom JavaScript

// Copy code button enhancement
document.addEventListener('DOMContentLoaded', function() {
  // Add any custom JavaScript functionality here

  // Example: Add external link icons
  document.querySelectorAll('a[href^="http"]').forEach(function(link) {
    if (!link.querySelector('.twemoji')) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
    }
  });
});

// Version selector enhancement (if using mike)
document.addEventListener('DOMContentLoaded', function() {
  const selector = document.querySelector('.md-header__topic');
  if (selector) {
    // Version selector customizations can go here
  }
});
