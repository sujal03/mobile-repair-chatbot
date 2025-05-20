$(document).ready(function () {
  console.log("Document is ready");
  const $chatMessages = $("#chat-messages");
  const $form = $("#chat-form");
  const $input = $("#user_message");
  const $sendBtn = $("#send-btn");
  const $suggestionsDesktop = $("#suggestions-desktop");
  const $suggestionsMobile = $("#suggestions-mobile");
  const $imageInput = $("#image-upload");
  const $imagePreview = $("#image-preview");

  // Clear local storage on page load to prevent reloading old chat
  localStorage.removeItem("chatHistory");
  $chatMessages.empty(); // Clear any existing messages on the frontend

  // Reset the session on page load
  $.ajax({
    type: "POST",
    url: "/reset",
    xhrFields: { withCredentials: true },
    success: function (response) {
      if (response.status === "success") {
        console.log("Chat session reset on page load");
        // Clear chat messages
        $chatMessages.empty();
        // Update suggestions after reset
        updateSuggestions("");
      } else {
        console.error("Failed to reset chat on page load:", response.error);
      }
    },
    error: function (xhr) {
      console.error("Error resetting chat on page load:", xhr.responseText);
    },
  });

  // Update suggestions based on user input
  function updateSuggestions(userMessage) {
    $.ajax({
      type: "POST",
      url: "/get_suggestions",
      contentType: "application/json",
      data: JSON.stringify({ user_message: userMessage || "" }),
      xhrFields: { withCredentials: true },
      success: function (response) {
        const suggestions = response.suggestions || [
          "What is the cost of screen replacement?",
          "Want to compare two devices?",
          "Do you repair water-damaged phones?",
        ];
        $suggestionsDesktop
          .empty()
          .append(
            suggestions
              .map((s) => `<div class="suggestion-chip">${s}</div>`)
              .join("")
          );
        $suggestionsMobile
          .empty()
          .append(
            suggestions
              .map((s) => `<div class="suggestion-chip">${s}</div>`)
              .join("")
          );
      },
      error: function (xhr) {
        console.error("Error fetching suggestions:", xhr.responseText);
        const defaults = [
          "What is the cost of screen replacement?",
          "Want to compare two devices?",
          "Do you repair water-damaged phones?",
        ];
        $suggestionsDesktop
          .empty()
          .append(
            defaults
              .map((s) => `<div class="suggestion-chip">${s}</div>`)
              .join("")
          );
        $suggestionsMobile
          .empty()
          .append(
            defaults
              .map((s) => `<div class="suggestion-chip">${s}</div>`)
              .join("")
          );
      },
    });
  }

  // Handle suggestion clicks
  $(".suggestions, .suggestions-mobile").on(
    "click",
    ".suggestion-chip",
    function () {
      $input.val($(this).text());
      $imageInput.val("");
      $imagePreview.hide();
      $form.submit();
    }
  );

  // Image upload preview
  $imageInput.on("change", function (e) {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        alert("Please upload an image file.");
        $imageInput.val("");
        $imagePreview.hide();
        return;
      }
      if (file.size > 5 * 1024 * 1024) {
        alert("Image size must be less than 5MB.");
        $imageInput.val("");
        $imagePreview.hide();
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => $imagePreview.attr("src", e.target.result).show();
      reader.readAsDataURL(file);
    } else {
      $imagePreview.hide();
    }
  });

  // Form submission
  $form.submit(function (e) {
    e.preventDefault();
    const userMessage = $input.val().trim();
    const imageFile = $imageInput[0].files[0];
    if (!userMessage && !imageFile) return;

    const formData = new FormData();
    formData.append("user_message", userMessage);
    if (imageFile) formData.append("image", imageFile);

    const now = new Date();
    let messageHtml = '<div class="message user-message">';
    if (imageFile)
      messageHtml += `<img src="${$imagePreview.attr(
        "src"
      )}" alt="User uploaded image" class="message-image">`;
    if (userMessage)
      messageHtml += `<div class="message-content"><div class="message-text">${userMessage}</div></div>`;
    messageHtml += `<div class="message-timestamp">${formatTime(
      now
    )}</div></div>`;
    $chatMessages.append(messageHtml);

    $chatMessages.append(
      '<div class="message bot-message" id="loading">' +
        '<div class="bot-icon"><i class="fas fa-robot"></i></div>' +
        '<div class="loading-dots"><span></span><span></span><span></span></div>' +
        "</div>"
    );
    $chatMessages.scrollTop($chatMessages[0].scrollHeight);
    $sendBtn.prop("disabled", true);

    $.ajax({
      type: "POST",
      url: "/chatbot",
      data: formData,
      processData: false,
      contentType: false,
      xhrFields: { withCredentials: true },
      success: function (response) {
        $("#loading").remove();
        const sanitizedResponse = DOMPurify.sanitize(
          marked.parse(response.bot_response)
        );
        $chatMessages.append(
          '<div class="message bot-message">' +
            '<div class="bot-icon"><i class="fas fa-robot"></i></div>' +
            "<div>" +
            '<div class="message-content"><div class="message-text">' +
            sanitizedResponse +
            "</div></div>" +
            '<div class="message-timestamp">' +
            formatTime(new Date(response.timestamp)) +
            "</div>" +
            "</div></div>"
        );
        $chatMessages.scrollTop($chatMessages[0].scrollHeight);
        localStorage.setItem("chatHistory", $chatMessages.html());
        updateSuggestions(userMessage);
      },
      error: function (xhr) {
        console.error("Error in chatbot request:", xhr.responseText);
        $("#loading").remove();
        $chatMessages.append(
          '<div class="message bot-message">' +
            '<div class="bot-icon"><i class="fas fa-robot"></i></div>' +
            "<div>" +
            '<div class="message-content"><div class="message-text">Sorry, something went wrong. Try again.</div></div>' +
            '<div class="message-timestamp">' +
            formatTime(new Date()) +
            "</div>" +
            "</div></div>"
        );
        $chatMessages.scrollTop($chatMessages[0].scrollHeight);
        localStorage.setItem("chatHistory", $chatMessages.html());
        updateSuggestions(userMessage);
      },
      complete: function () {
        $sendBtn.prop("disabled", false);
        $input.val("");
        $imageInput.val("");
        $imagePreview.hide();
      },
    });
  });

  // Handle refresh chat button click
  $("#refresh-chat-btn").on("click", function () {
    $.ajax({
      type: "POST",
      url: "/reset",
      xhrFields: { withCredentials: true },
      success: function (response) {
        if (response.status === "success") {
          // Clear chat messages
          $chatMessages.empty();
          // Clear local storage
          localStorage.removeItem("chatHistory");
          // Update suggestions after reset
          updateSuggestions("");
        } else {
          alert("Failed to reset chat: " + (response.error || "Unknown error"));
        }
      },
      error: function (xhr) {
        console.error("Error resetting chat:", xhr.responseText);
        alert("Failed to reset chat. Please try again.");
      },
    });
  });

  // Format timestamp
  function formatTime(date) {
    return date.toLocaleString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
      day: "2-digit",
      month: "short",
    });
  }

  // Fade out flash messages (if any)
  setTimeout(() => {
    $(".alert")
      .fadeTo(500, 0)
      .slideUp(500, function () {
        $(this).remove();
      });
  }, 5000);

  // Initialize suggestions (already called after reset)
});
