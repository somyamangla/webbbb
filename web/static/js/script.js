let list = document.querySelector(".bottom-bar__list");

let activeItemIndex = 1;

let items = list.children;

const handleClick = (index) => {
  if (index !== activeItemIndex) {
    items[activeItemIndex].classList.remove("selected");
    items[index].classList.add("selected");

    let direction;
    index - activeItemIndex > 0 ? (direction = 1) : (direction = -1);

    let magnitude = Math.abs(index - activeItemIndex);

    activeItemIndex = index;

    items[0].style.transform =
      "translate(" + (activeItemIndex - 1) * 100 + "%, -0.5rem)";

    items[0].classList.add("active--" + magnitude);
    items[0].classList.add(direction > 0 ? "active-right" : "active-left");
    console.log(items[0].classList);

    setTimeout(() => {
      items[0].classList.remove("active--" + magnitude);
      items[0].classList.remove(direction > 0 ? "active-right" : "active-left");
    }, 200);
  }
};

Object.keys(items).forEach((item, index) => {
  items[index].addEventListener("click", () => {
    handleClick(index);
  });
});