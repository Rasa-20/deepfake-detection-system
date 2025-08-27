import React from "react"; /* enables the JSX and component functionality */
import { useLocation } from "react-router-dom";  /* to access pages */
import DefaultLayout from "../../layouts/DefaultLayout";  /* result page with the header & footer */

const Results = () => {
  const location = useLocation();
  const result = location.state || JSON.parse(localStorage.getItem("analysisResult"));

  if (!result) {
    return (
      <DefaultLayout>
        <div className="flex items-center justify-center min-h-screen">
          <p className="bg-red-400 text-white px-6 py-4 rounded-lg shadow-lg">
            No analysis result found. Please upload a video first.
          </p>
        </div>
      </DefaultLayout>
    );
  }

  const { status, confidence, isFake } = result;

  return (
    <DefaultLayout>
      <div className="bg-gray-400 min-h-screen py-20 px-6 flex flex-col items-center text-white">
        <div className="bg-cyan-50 text-gray-900 rounded-2xl p-10 w-full max-w-2xl shadow-2xl space-y-10">

          <h1 className="text-4xl font-extrabold text-center">Analysis Results</h1>

          <div className={`flex items-center gap-4 p-6 rounded-xl shadow-inner 
            ${isFake 
              ? "bg-gradient-to-r from-rose-100 to-rose-200 border border-rose-300" 
              : "bg-gradient-to-r from-emerald-100 to-emerald-200 border border-emerald-300"
            }`}>

            <div className={`w-14 h-14 flex items-center justify-center rounded-full text-3xl shadow 
              ${isFake 
                ? "bg-rose-500 text-white" 
                : "bg-emerald-500 text-white"
              }`}>
              {isFake ? "✖" : "✔"}
            </div>

            <div>
              <p className="text-2xl font-bold">{status}</p>
              <p className={`text-lg mt-1 ${isFake ? "text-rose-600" : "text-emerald-600"}`}>
                {confidence}% Confidence
              </p>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-bold mb-4">Detailed Analysis</h2>
            <ul className="space-y-2 text-gray-700">
  <li><strong>Visual Analysis:</strong> {result.visualAnalysis}</li>
  <li><strong>Audio Analysis:</strong> {result.audioAnalysis}</li>
  <li><strong>Metadata Analysis:</strong> {result.metadataAnalysis}</li>
</ul>

          </div>

          <div>
            <p className="text-gray-700 mb-2">Analysis Confidence</p>
            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
              <div className={`h-4 ${isFake ? "bg-rose-500" : "bg-emerald-500"}`} style={{ width: `${confidence}%` }}></div>
            </div>
            <p className="text-gray-700 mt-1 text-right">{confidence}%</p>
          </div>

          <div className="text-center pt-6">
            <a
              href="/"
              className="inline-block bg-gray-900 text-white hover:bg-gray-700 px-8 py-3 rounded-full transition shadow-lg"
            >
              Analyze Another Video
            </a>
          </div>

        </div>
      </div>
    </DefaultLayout>
  );
};

export default Results;
