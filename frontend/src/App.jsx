import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/home';
import Upload from './pages/upload';
import Results from './pages/results';
import Info from './pages/info';

const App = () => {                                                   /* URLs for pages */
  return (
    <Router>                                                
      <Routes>
        <Route path="/" element={<Home />} />               
        <Route path="/upload" element={<Upload />} />
        <Route path="/results" element={<Results />} />
        <Route path="/info" element={<Info />} />
      </Routes>
    </Router>
  );
};

export default App;
