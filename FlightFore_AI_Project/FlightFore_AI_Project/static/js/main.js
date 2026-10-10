document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll(".menu-link");

    buttons.forEach((button) => {
        button.addEventListener("click", () => {
            buttons.forEach((item) => item.classList.remove("clicked"));
            button.classList.add("clicked");
        });
    });
});
