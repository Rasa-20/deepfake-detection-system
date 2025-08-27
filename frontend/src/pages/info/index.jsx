import React from "react";
import DefaultLayout from "../../layouts/DefaultLayout";

const Info = () => {
    return(
        <DefaultLayout>
            {/* How the app works Section */}
      <div className="bg-gray-100 py-20 px-6">
        <h2 className="text-4xl font-extrabold text-center text-gray-800 mb-16">
          How It Works
        </h2>

        <div className="flex justify-center gap-12 flex-wrap">
          {/* Step 1: Upload */}
          <div className="bg-white rounded-xl shadow-lg p-8 w-80 text-center hover:shadow-2xl transition">
            <div className="text-5xl mb-4">📤</div>
            <h3 className="text-xl font-semibold mb-2">1. Upload Video</h3>
            <p className="text-gray-600">
              Users upload a video clip suspected to be fake. The app supports MP4, AVI, and other standard formats.
            </p>
          </div>

          {/* Step 2: Analysis */}
          <div className="bg-white rounded-xl shadow-lg p-8 w-80 text-center hover:shadow-2xl transition">
            <div className="text-5xl mb-4">🧠</div>
            <h3 className="text-xl font-semibold mb-2">2. AI Model Analysis</h3>
            <p className="text-gray-600">
              The video is processed using audio and spatio-temporal deep learning models to detect signs of manipulation.
            </p>
          </div>

          {/* Step 3: Result */}
          <div className="bg-white rounded-xl shadow-lg p-8 w-80 text-center hover:shadow-2xl transition">
            <div className="text-5xl mb-4">✅</div>
            <h3 className="text-xl font-semibold mb-2">3. Get Results</h3>
            <p className="text-gray-600">
              A confidence score and detection result are displayed, helping you verify authenticity within seconds.
            </p>
          </div>
        </div>
      </div>
        </DefaultLayout>
    )
};

export default Info;