import React from "react";
export default function AlertsPanel({ alerts }) {
  return (
    <div className="p-4 bg-white rounded-xl shadow">
      <h2 className="font-semibold mb-2">Alerts</h2>
      {alerts.map((a, i) => (
        <div
          key={i}
          className={`p-2 mb-2 rounded ${
            a.severity === "danger"
              ? "bg-red-100"
              : a.severity === "warn"
              ? "bg-yellow-100"
              : "bg-green-100"
          }`}
        >
          <div className="font-medium">{a.message}</div>
          <div className="text-sm">{a.detail}</div>
        </div>
      ))}
    </div>
  );
}