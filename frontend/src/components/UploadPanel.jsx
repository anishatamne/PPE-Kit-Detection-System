import React from "react";

export default function UploadPanel({ onUpload }) {
  return (
    <div className="p-6 bg-white shadow rounded-xl">
      <input
        type="file"
        accept="image/*"
        onChange={(e) => onUpload(e.target.files[0])}
        className="block w-full text-sm"
      />
    </div>
  );
}