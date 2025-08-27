import React from "react";
import { Link } from "react-router-dom"; // <- Make sure to import Link
import DefaultLayout from "../../layouts/DefaultLayout";

const Home = () => {
  return (
    <DefaultLayout>
      {/* Hero Section */}
      <div className="bg-gradient-to-b from-[#1C2841] to-[#00B8B8] text-white py-32 text-center">
        <h1 className="text-5xl font-extrabold mb-4">Detect Deepfakes with Confidence</h1>
        <p className="text-lg mb-8">Advanced AI-powered technology to identify manipulated videos with high accuracy</p>
        
        {/* Use Link instead of button */}
        <Link 
          to="/upload" 
          className="inline-block bg-black hover:bg-[#36454F] font-bold text-white px-6 py-3 rounded-full transition"
        >
          Analyze Video
        </Link>
      </div>

      {/* Why Choose Section */}
      <div className="bg-gray-300 py-20">
        <h2 className="text-3xl font-bold text-center mb-12 text-gray-800">Why Choose Our Technology</h2>

        <div className="flex justify-center gap-8 px-10 flex-wrap">
          <div className="bg-white shadow-lg rounded-lg p-6 w-72 text-center hover:shadow-xl transition">
            <h3 className="text-lg font-bold mb-2">96% Accuracy</h3>
            <p>State-of-the-art AI models trained on millions of videos.</p>
          </div>

          <div className="bg-white shadow-lg rounded-lg p-6 w-72 text-center hover:shadow-xl transition">
            <h3 className="text-lg font-bold mb-2">Real-time Analysis</h3>
            <p>Get results in seconds, not minutes.</p>
          </div>

          <div className="bg-white shadow-lg rounded-lg p-6 w-72 text-center hover:shadow-xl transition">
            <h3 className="text-lg font-bold mb-2">Detailed Reports</h3>
            <p>Comprehensive analysis with frame-by-frame breakdown.</p>
          </div>
        </div>
      </div>
    </DefaultLayout>
  );
};

export default Home;
