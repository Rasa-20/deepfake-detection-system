import React from 'react';
import Header from '../components/Header';  /* Header and Footer components from the components folder */
import Footer from '../components/Footer';

const DefaultLayout = ({ children }) => {
  return (
    <div className="flex flex-col min-h-screen">      
      <Header />
      <main className="flex-grow">{children}</main>
      <Footer />
    </div>
  );
};

export default DefaultLayout;
