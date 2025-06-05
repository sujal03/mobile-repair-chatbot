$(document).ready(function () {
  let conversations = [];
  let sortField = "latest_timestamp";
  let sortDirection = "desc";
  let sortCache = {};

  // Load analytics
  function loadAnalytics() {
    $.ajax({
      url: "/api/admin/analytics",
      method: "GET",
      xhrFields: { withCredentials: true },
      success: function (data) {
        if (data.status === "success") {
          $("#total-conversations").text(data.analytics.total_conversations);
          $("#total-messages").text(data.analytics.total_messages);
          $("#avg-messages").text(data.analytics.avg_messages_per_session);
        }
      },
      error: function (xhr) {
        console.error("Error loading analytics:", xhr.responseJSON?.message);
      },
    });
  }

  // Load all conversations
  function loadConversations() {
    $(".loading-spinner").show();
    $.ajax({
      url: "/api/admin/conversations",
      method: "GET",
      xhrFields: { withCredentials: true },
      success: function (data) {
        $(".loading-spinner").hide();
        const tbody = $("#conversation-table");
        tbody.empty();
        if (data.status === "success" && data.conversations.length > 0) {
          conversations = data.conversations;
          sortConversations();
          renderConversations();
        } else {
          tbody.append(`
                                <tr>
                                    <td colspan="6" class="empty-message">No conversations found</td>
                                </tr>
                            `);
        }
      },
      error: function (xhr) {
        $(".loading-spinner").hide();
        $("#conversation-table").html(`
                            <tr>
                                <td colspan="6" class="empty-message">Error loading conversations: ${
                                  xhr.responseJSON?.message ||
                                  "Please try again"
                                }</td>
                            </tr>
                        `);
      },
    });
  }

  // Sort conversations
  function sortConversations() {
    const cacheKey = `${sortField}_${sortDirection}`;
    if (sortCache[cacheKey]) {
      conversations = sortCache[cacheKey];
      return;
    }

    conversations.sort((a, b) => {
      let valA = a[sortField] || "";
      let valB = b[sortField] || "";

      if (sortField === "latest_timestamp") {
        valA = valA ? new Date(valA).getTime() : 0;
        valB = valB ? new Date(valB).getTime() : 0;
      } else if (sortField === "message_count") {
        valA = parseInt(valA) || 0;
        valB = parseInt(valB) || 0;
      } else {
        valA = valA.toString().toLowerCase();
        valB = valB.toString().toLowerCase();
      }

      if (sortDirection === "asc") {
        return valA > valB ? 1 : valA < valB ? -1 : 0;
      } else {
        return valA < valB ? 1 : valA > valB ? -1 : 0;
      }
    });

    sortCache[cacheKey] = [...conversations];
  }

  // Render all conversations
  function renderConversations() {
    const tbody = $("#conversation-table");
    tbody.empty();
    conversations.forEach((conv) => {
      const roleClass =
        conv.latest_role === "user" ? "badge-user" : "badge-bot";
      tbody.append(`
                        <tr data-session-id="${conv.session_id}">
                            <td><input type="checkbox" class="select-conversation" value="${
                              conv.session_id
                            }"></td>
                            <td><span class="message-preview" title="${
                              conv.session_id || "N/A"
                            }">${conv.session_id || "N/A"}</span></td>
                            <td><span class="message-preview" title="${
                              conv.latest_message || "N/A"
                            }">${conv.latest_message || "N/A"}</span></td>
                            <td><span class="badge ${roleClass}">${
        conv.latest_role || "N/A"
      }</span></td>
                            <td><span class="message-preview" title="${
                              conv.latest_timestamp
                                ? new Date(
                                    conv.latest_timestamp
                                  ).toLocaleString()
                                : "N/A"
                            }">${
        conv.latest_timestamp
          ? new Date(conv.latest_timestamp).toLocaleString()
          : "N/A"
      }</span></td>
                            <td>${conv.message_count || 0}</td>
                        </tr>
                    `);
    });
    updateDeleteButton();
  }

  // Update delete button state
  function updateDeleteButton() {
    const checked = $(".select-conversation:checked").length;
    $("#delete-selected").prop("disabled", checked === 0);
  }

  // Handle sorting
  $(".sortable").click(function () {
    const newSortField = $(this).data("sort");
    if (newSortField === sortField) {
      sortDirection = sortDirection === "asc" ? "desc" : "asc";
    } else {
      sortField = newSortField;
      sortDirection = "asc";
    }

    $(".sortable").removeClass("sort-asc sort-desc sorted");
    $(this).addClass(`sort-${sortDirection} sorted`);

    sortConversations();
    renderConversations();
  });

  // Load conversation details
  $("#conversation-table").on(
    "click",
    "tr[data-session-id] td:not(:first-child)",
    function () {
      const sessionId = $(this).parent().data("session-id");
      $(".loading-spinner").show();
      $.ajax({
        url: `/api/admin/conversation/${sessionId}`,
        method: "GET",
        xhrFields: { withCredentials: true },
        success: function (data) {
          $(".loading-spinner").hide();
          if (data.status === "success") {
            const messagesDiv = $("#conversation-messages");
            messagesDiv.empty();
            data.messages.forEach((msg) => {
              const isUser = msg.role === "user";
              const messageContent = isUser
                ? msg.message
                : marked.parse(msg.message);
              const sanitizedContent = DOMPurify.sanitize(messageContent);
              messagesDiv.append(`
                                    <div class="message ${
                                      isUser ? "user-message" : "bot-message"
                                    }">
                                        <div class="message-header">
                                            <span>${
                                              msg.role.charAt(0).toUpperCase() +
                                              msg.role.slice(1)
                                            }</span>
                                            <button class="copy-btn" title="Copy Message"><i class="fas fa-copy"></i></button>
                                        </div>
                                        <div class="message-content">${sanitizedContent}</div>
                                        <div class="timestamp">${new Date(
                                          msg.timestamp
                                        ).toLocaleString()}</div>
                                    </div>
                                `);
            });
            $("#conversationModalLabel").text(`Conversation Details`);
            $("#conversationModalSubtitle").text(`Session: ${sessionId}`);
            $("#conversationModal").modal("show");
            // Scroll to the bottom of the modal body
            const modalBody = $("#conversation-messages")[0];
            modalBody.scrollTop = modalBody.scrollHeight;
          } else {
            alert("Error loading conversation: " + data.message);
          }
        },
        error: function (xhr) {
          $(".loading-spinner").hide();
          alert(
            "Error loading conversation: " +
              (xhr.responseJSON?.message || "Please try again")
          );
        },
      });
    }
  );

  // Handle copy button
  $("#conversation-messages").on("click", ".copy-btn", function () {
    const messageContent = $(this)
      .closest(".message")
      .find(".message-content")
      .text();
    navigator.clipboard
      .writeText(messageContent)
      .then(() => {
        $(this).html('<i class="fas fa-check"></i>').css("color", "#22c55e");
        setTimeout(() => {
          $(this).html('<i class="fas fa-copy"></i>').css("color", "#6b7280");
        }, 1000);
      })
      .catch(() => {
        alert("Failed to copy message");
      });
  });

  // Handle select all checkbox
  $("#select-all").change(function () {
    $(".select-conversation").prop("checked", this.checked);
    updateDeleteButton();
  });

  // Handle individual checkbox changes
  $("#conversation-table").on(
    "change",
    ".select-conversation",
    updateDeleteButton
  );

  // Handle delete selected conversations
  $("#delete-selected").click(function () {
    const sessionIds = $(".select-conversation:checked")
      .map(function () {
        return $(this).val();
      })
      .get();
    if (sessionIds.length === 0) return;
    if (
      !confirm(
        `Are you sure you want to delete ${sessionIds.length} conversation(s)?`
      )
    )
      return;

    $(".loading-spinner").show();
    $.ajax({
      url: "/api/admin/conversations/delete",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ session_ids: sessionIds }),
      xhrFields: { withCredentials: true },
      success: function (data) {
        $(".loading-spinner").hide();
        if (data.status === "success") {
          alert(data.message);
          window.location.reload();
        } else {
          alert("Error deleting conversations: " + data.message);
        }
      },
      error: function (xhr) {
        $(".loading-spinner").hide();
        alert(
          "Error deleting conversations: " +
            (xhr.responseJSON?.message || "Please try again")
        );
      },
    });
  });

  // Initial load
  loadAnalytics();
  loadConversations();
});
