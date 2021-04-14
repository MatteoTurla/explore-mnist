// Code goes here

var canvas = new fabric.Canvas(document.getElementById("c"), {
  isDrawingMode: true,
});

var brush = new fabric.PencilBrush(canvas);

brush.onMouseDown({ x: points[0][0], y: points[0][1] });
for (var i = 1; i < points.length; i++) {
  brush.onMouseMove({ x: points[i][0], y: points[i][1] });
}

function save() {
  document.getElementById("c").toBlob(function (blob) {
    saveAs(blob, "myIMG.png");
  });
}
