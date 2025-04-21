var open_btn = document.querySelector(".open-btn");
var chat_box = document.querySelector(".chatbot_as");
var close_btn=document.querySelector(".btn-exit-mb");
var chat_input=document.querySelector(".ch-input .putin");
var form=document.querySelector(".ch-input");
var msg=document.querySelector(".chat-message");
var check = 1;

open_btn.addEventListener("click", function() {
    if (check === 0) {
        open_btn.style.background="green";
        chat_box.classList.remove("off_chat");
        chat_box.classList.add("chatbot_as");
        check = 1;
    } else {
        chat_box.classList.remove("chatbot_as");
        chat_box.classList.add("off_chat");
        open_btn.style.background="red";
        check = 0;
    }
});

close_btn.addEventListener("click",function(){
    if(check===1)
    {
        chat_box.classList.remove("chatbot_as");
        chat_box.classList.add("off_chat");
        open_btn.style.background = "red";
        check = 0;
    }
})


// form.addEventListener('submit', function(event) {
//     // Prevent the form from submitting the traditional way
//     event.preventDefault();

//     // Get the input element using querySelector
//     const input = document.querySelector('.putin');

//     // Get the value from the input element
//     const inputValue = input.value;

//     // Log the value to the console
//     console.log('Input Value:', inputValue);
//     input.value=" ";
//     const messageElement = document.createElement('li');
//     const AImessage=document.createElement("li");
//     messageElement.textContent = inputValue;
//     AImessage.textContent = "Loading...";
//     messageElement.classList.add("outgoing")
//     AImessage.classList.add("incoming");
//     messageElement.style.clear = 'both';
//     AImessage.style.clear = 'both';
//     msg.appendChild(messageElement);
//     msg.appendChild(AImessage);
// });

document.getElementsByClassName("lang-right").addEventListener("click", function() {
    var dropdown = document.getElementById("dropdown-content");
    if (dropdown.style.display === "none" || dropdown.style.display === "") {
        dropdown.style.display = "block";
    } else {
        dropdown.style.display = "none";
    }
});

// Close the dropdown when the user clicks outside of it
window.addEventListener("click", function(event) {
    var dropdown = document.getElementById("dropdown-content");
    var button = document.getElementById("lang-right");
    if (event.target !== button && event.target !== dropdown) {
        dropdown.style.display = "none";
    }
});


