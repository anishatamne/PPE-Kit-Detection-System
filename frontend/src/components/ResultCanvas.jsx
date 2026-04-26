import React, { useEffect, useRef } from "react";

export default function ResultCanvas({ image, data }) {
  const canvasRef = useRef();

  useEffect(() => {
    if (!image || !data || !data.persons) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const img = new Image();
    img.src = URL.createObjectURL(image);

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      data.persons.forEach((p) => {
        if (!p.bbox || p.bbox.length !== 4) return;

        const [x1, y1, x2, y2] = p.bbox;

        ctx.strokeStyle = p.compliant ? "green" : "red";
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = "yellow";
        ctx.fillText(`ID ${p.id}`, x1, y1 - 5);
      });

      // ✅ FIXED keypoints access
      data.persons.forEach((p) => {
        if (!p.pose_valid || !p.keypoints) return;

        p.keypoints.forEach((k) => {
          ctx.beginPath();
          ctx.arc(k.x, k.y, 3, 0, 2 * Math.PI);
          ctx.fillStyle = "cyan";
          ctx.fill();
        });
      });
    };
  }, [image, data]);

  return (
    <canvas ref={canvasRef} className="border rounded-xl shadow" />
  );
}