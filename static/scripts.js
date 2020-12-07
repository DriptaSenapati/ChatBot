function speak(text) {
  if ("speechSynthesis" in window) {
    var msg = new SpeechSynthesisUtterance();
    var voices = window.speechSynthesis.getVoices();
    msg.voice = voices[0];
    // msg.rate = $('#rate').val() / 10;
    // msg.pitch = $('#pitch').val();
    msg.text = text;
    speechSynthesis.speak(msg);
  } else {
    console.log("No Support");
  }
}
try {
  var SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = new SpeechRecognition();
} catch (e) {
  console.error(e);
}

var noteTextarea = $("#textarea");
var noteContent = "";

recognition.continuous = false;

recognition.onresult = function (event) {
  var current = event.resultIndex;
  var transcript = event.results[current][0].transcript;
  var mobileRepeatBug =
    current == 1 && transcript == event.results[0][0].transcript;
  if (!mobileRepeatBug) {
    noteContent += transcript;
    noteTextarea.val(noteContent);
    $.post(
      "/call",
      {
        question: noteContent,
      },
      function (data, status, xhr) {
        noteTextarea.val(data);
        speak(data);
      }
    );
  }
};

recognition.onspeechend = function () {
  $(".controls").removeClass("activate");
};

// recognition.onerror = function (event) {
//   if (event.error == "no-speech") {
//     instructions.text("No speech was detected. Try again.");
//   }
// };

$("#startRecogniton").on("click", function (e) {
  noteTextarea.val("Listening...");
  if (noteContent.length) {
    noteContent = "";
  }
  recognition.start();
  $(".controls").addClass("activate");
});
$(".btn-outline-dark").on("click", function (e) {
  noteTextarea.val("");
});
$(".Cancel").on("click", function (e) {
  recognition.stop();
  noteTextarea.val("");
  $(".controls").removeClass("activate");
});

noteTextarea.on("input", function () {
  noteContent = $(this).val();
});

noteTextarea.on("change", function () {
  noteContent = $(this).val();
  $.post(
    "/call",
    {
      question: noteContent,
    },
    function (data, status, xhr) {
      noteTextarea.val(data);
      speak(data);
    }
  );
});
