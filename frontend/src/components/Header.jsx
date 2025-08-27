import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="bg-gray-800 shadow p-4">
      <div className="container mx-auto flex justify-between items-center">
        <h1 className="text-[#20B2AA] font-bold text-2xl">DeepScan</h1>
        <nav className="space-x-6">
          <Link to="/" className="text-gray-300 text-xl font-bold hover:text-[#20B2AA] ">Home</Link>
          <Link to="/info" className="text-gray-300 text-xl font-bold hover:text-[#20B2AA]">How It Works</Link>
          <Link to="/upload" className="text-gray-300 text-xl font-bold hover:text-[#20B2AA]">Upload</Link>
          <Link to="/results" className="text-gray-300 text-xl font-bold hover:text-[#20B2AA]">Results</Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
