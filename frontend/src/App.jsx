import React, { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import ResultCanvas from "./components/ResultCanvas";
import AlertsPanel from "./components/AlertsPanel";
import { analyzeImage } from "./api";

export default function App() {
  const [image, setImage] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (file) => {
    setImage(file);
    setLoading(true);

    const res = await analyzeImage(file);

    if (!res) {
        alert("API failed. Check backend.");
        setLoading(false);
        return;
    }
    console.log("API RESPONSE:", res);
    setData(res);
    setLoading(false);
    };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto space-y-6">

        <h1 className="text-3xl font-bold">PPE Compliance Monitor</h1>

        <UploadPanel onUpload={handleUpload} />

        {loading && <p>Processing...</p>}

        {image && data && (
          <div className="grid grid-cols-2 gap-6">
            <ResultCanvas image={image} data={data} />
            <AlertsPanel alerts={data.alerts} />
          </div>
        )}
      </div>
    </div>
  );
}