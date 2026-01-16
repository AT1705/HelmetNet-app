import { Shield } from 'lucide-react';

interface NavigationProps {
  currentPage: string;
  onNavigate: (page: string) => void;
}

export function Navigation({ currentPage, onNavigate }: NavigationProps) {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-sm border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => onNavigate('home')}>
            <Shield className="size-8 text-slate-700" />
            <span className="font-bold text-xl text-slate-900">HelmetNet</span>
          </div>
          <div className="flex gap-8 items-center">
            <button
              onClick={() => onNavigate('home')}
              className={`transition-colors font-medium ${
                currentPage === 'home'
                  ? 'text-slate-900 font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Home
            </button>
            <button
              onClick={() => onNavigate('about')}
              className={`transition-colors font-medium ${
                currentPage === 'about'
                  ? 'text-slate-900 font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              About
            </button>
            <button
              onClick={() => onNavigate('demo')}
              className={`px-6 py-2.5 rounded-xl transition-all font-semibold shadow-md hover:shadow-lg ${
                currentPage === 'demo'
                  ? 'bg-amber-500 text-slate-900'
                  : 'bg-amber-500 text-slate-900 hover:bg-amber-400'
              }`}
            >
              Start Demo
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
