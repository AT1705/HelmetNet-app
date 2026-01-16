import { useState } from 'react';
import { Navigation } from '@/app/components/Navigation';
import { HomePage } from '@/app/components/HomePage';
import { AboutPage } from '@/app/components/AboutPage';
import { DemoPage } from '@/app/components/DemoPage';

type Page = 'home' | 'about' | 'demo';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');

  const handleNavigate = (page: string) => {
    setCurrentPage(page as Page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <Navigation currentPage={currentPage} onNavigate={handleNavigate} />
      
      {currentPage === 'home' && <HomePage onNavigate={handleNavigate} />}
      {currentPage === 'about' && <AboutPage />}
      {currentPage === 'demo' && <DemoPage />}
    </div>
  );
}