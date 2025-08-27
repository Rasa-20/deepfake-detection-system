import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import DefaultLayout from "../../layouts/DefaultLayout";

const Upload = () => {
  const [video, setVideo] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    // ✅ Accept only .mp4 files
    if (selectedFile && selectedFile.type !== "video/mp4") {
      alert("Only .mp4 files are allowed.");
      return;
    }

    setVideo(selectedFile);
  };

  const handleAnalyze = async () => {
    if (!video) {
      alert("Please select a video file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", video);

    try {
      const response = await fetch("localhost url", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.error) {
        // ✅ Show backend error if any (e.g., no audio, wrong file)
        alert(`Error: ${data.error}`);
        return;
      }

      localStorage.setItem("analysisResult", JSON.stringify(data));
      navigate("/results");

    } catch (error) {
      console.error("Error analyzing video:", error);
      alert("Failed to analyze video. Please try again.");
    }
  };

  return (
    <DefaultLayout>
      <div className="bg-gray-300 min-h-screen py-20 flex flex-col items-center">
        <h1 className="text-4xl font-bold mb-12 text-center">Upload Video for Analysis</h1>

        <div className="bg-white shadow-lg rounded-lg p-10 w-full max-w-3xl">
          <h2 className="text-2xl font-semibold mb-4">Video Upload</h2>
          <p className="text-gray-600 mb-8">
            Upload an .mp4 file to analyze it for potential deepfake manipulation.
          </p>

          {!video && (
            <>
              <label
                htmlFor="videoUpload"
                className="border-2 border-dashed border-gray-300 rounded-lg p-10 flex flex-col items-center justify-center hover:border-teal-400 transition mb-8 cursor-pointer"
              >
                <p className="text-gray-700 mb-4 font-medium">Drag and drop your video here</p>
                <div className="bg-gradient-to-b from-cyan-500 to-teal-500 hover:from-teal-600 hover:to-cyan-600 text-white px-6 py-3 rounded-full transition">
                  Select Video
                </div>
                <p className="text-gray-400 mt-4">or click here to browse files</p>
              </label>

              <input
                id="videoUpload"
                type="file"
                accept="video/mp4"
                className="hidden"
                onChange={handleFileChange}
              />
            </>
          )}

          {video && (
            <div className="mt-4 text-center">
              <video
                src={URL.createObjectURL(video)}
                controls
                className="w-full max-w-md mx-auto rounded-md shadow"
              />
              <p className="text-sm text-gray-700 mt-2">
                Selected file: <span className="font-semibold">{video.name}</span>
              </p>
            </div>
          )}

          <div className="text-center mt-6">
            <button
              onClick={handleAnalyze}
              className="bg-black text-white px-8 py-3 rounded-full hover:bg-gray-700 transition"
            >
              Analyze Video
            </button>
          </div>
        </div>
      </div>
    </DefaultLayout>
  );
};

export default Upload;
