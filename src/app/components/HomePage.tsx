import { Shield, Camera, AlertCircle, TrendingUp, CheckCircle, Zap, Users } from 'lucide-react';
import { ImageWithFallback } from '@/app/components/figma/ImageWithFallback';

interface HomePageProps {
  onNavigate: (page: string) => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  const features = [
    {
      icon: Camera,
      title: 'Real-Time Detection',
      description: 'Advanced AI algorithms detect helmet compliance in milliseconds with 99.2% accuracy.'
    },
    {
      icon: Shield,
      title: 'Enhanced Safety',
      description: 'Automated monitoring ensures consistent enforcement of helmet safety regulations.'
    },
    {
      icon: Zap,
      title: 'Instant Alerts',
      description: 'Immediate notifications to traffic authorities when violations are detected.'
    },
    {
      icon: TrendingUp,
      title: 'Analytics Dashboard',
      description: 'Comprehensive insights into compliance rates and traffic patterns over time.'
    }
  ];

  const stats = [
    { value: '99.2%', label: 'Detection Accuracy' },
    { value: '50K+', label: 'Daily Scans' },
    { value: '35%', label: 'Reduction in Violations' },
    { value: '24/7', label: 'Monitoring' }
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Hero Section */}
      <section className="relative h-[90vh] flex items-center">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900">
          <ImageWithFallback
            src="https://images.unsplash.com/photo-1645094118521-5c72e98985e5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb3RvcmN5Y2xlJTIwaGVsbWV0JTIwcmlkZXJ8ZW58MXx8fHwxNzY4NDgyNTU2fDA&ixlib=rb-4.1.0&q=80&w=1080"
            alt="Motorcycle rider with helmet"
            className="w-full h-full object-cover opacity-20"
          />
        </div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-white">
          <div className="max-w-3xl">
            <div className="flex items-center gap-2 mb-6">
              <Shield className="size-12" />
              <span className="text-sm font-semibold tracking-wider uppercase">AI-Powered Road Safety</span>
            </div>
            <h1 className="text-6xl font-bold mb-6">
              Protecting Lives Through Intelligent Helmet Detection
            </h1>
            <p className="text-xl mb-8 text-gray-200">
              HelmetNet uses cutting-edge artificial intelligence to automatically detect and monitor motorcycle helmet compliance, making roads safer for everyone.
            </p>
            <div className="flex gap-4">
              <button
                onClick={() => onNavigate('demo')}
                className="px-8 py-4 bg-amber-500 text-slate-900 rounded-xl font-semibold hover:bg-amber-400 transition-all shadow-xl hover:shadow-2xl hover:scale-105"
              >
                Try Demo Now
              </button>
              <button
                onClick={() => onNavigate('about')}
                className="px-8 py-4 border-2 border-white text-white rounded-xl font-semibold hover:bg-white/10 transition-all"
              >
                Learn More
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-4xl font-bold text-slate-800 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4 text-slate-900">Powerful Features for Road Safety</h2>
            <p className="text-xl text-gray-600">
              Advanced technology designed to save lives and improve traffic safety compliance
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div key={index} className="bg-white p-6 rounded-xl border border-slate-200 hover:border-slate-400 hover:shadow-xl transition-all duration-300">
                  <div className="size-12 bg-slate-100 rounded-xl flex items-center justify-center mb-4">
                    <Icon className="size-6 text-slate-700" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2 text-slate-900">{feature.title}</h3>
                  <p className="text-gray-600 text-sm">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4 text-slate-900">How HelmetNet Works</h2>
            <p className="text-xl text-gray-600">Simple, accurate, and reliable</p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="size-20 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl flex items-center justify-center mx-auto mb-6 text-white text-2xl font-bold shadow-lg">
                1
              </div>
              <h3 className="text-xl font-semibold mb-3 text-slate-900">Capture</h3>
              <p className="text-gray-600">
                High-resolution cameras continuously monitor traffic and capture motorcycle riders
              </p>
            </div>
            <div className="text-center">
              <div className="size-20 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl flex items-center justify-center mx-auto mb-6 text-white text-2xl font-bold shadow-lg">
                2
              </div>
              <h3 className="text-xl font-semibold mb-3 text-slate-900">Analyze</h3>
              <p className="text-gray-600">
                AI algorithms instantly detect and verify helmet presence with high accuracy
              </p>
            </div>
            <div className="text-center">
              <div className="size-20 bg-gradient-to-br from-slate-700 to-slate-800 rounded-xl flex items-center justify-center mx-auto mb-6 text-white text-2xl font-bold shadow-lg">
                3
              </div>
              <h3 className="text-xl font-semibold mb-3 text-slate-900">Alert</h3>
              <p className="text-gray-600">
                Automated notifications sent to authorities for immediate action on violations
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Impact Section */}
      <section className="py-20 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-4xl font-bold mb-6 text-slate-900">Making a Real Impact</h2>
              <p className="text-lg text-gray-600 mb-8">
                Cities implementing HelmetNet have seen significant improvements in road safety and helmet compliance rates.
              </p>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="size-6 text-green-500 flex-shrink-0 mt-1" />
                  <div>
                    <div className="font-semibold text-slate-900">Reduced Road Fatalities</div>
                    <div className="text-gray-600">28% decrease in motorcycle-related deaths in pilot cities</div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="size-6 text-green-500 flex-shrink-0 mt-1" />
                  <div>
                    <div className="font-semibold text-slate-900">Improved Compliance</div>
                    <div className="text-gray-600">Helmet usage increased from 65% to 91% within 6 months</div>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <CheckCircle className="size-6 text-green-500 flex-shrink-0 mt-1" />
                  <div>
                    <div className="font-semibold text-slate-900">Efficient Enforcement</div>
                    <div className="text-gray-600">Automated system reduces manual monitoring costs by 70%</div>
                  </div>
                </div>
              </div>
            </div>
            <div className="relative h-[500px] rounded-xl overflow-hidden shadow-2xl border border-slate-200">
              <ImageWithFallback
                src="https://images.unsplash.com/photo-1766524872202-ae69163a4218?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx0cmFmZmljJTIwc2FmZXR5JTIwY2FtZXJhfGVufDF8fHx8MTc2ODQ4MjU1N3ww&ixlib=rb-4.1.0&q=80&w=1080"
                alt="Traffic monitoring"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-slate-800 to-slate-900 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold mb-6">Ready to See It in Action?</h2>
          <p className="text-xl mb-8 text-gray-300">
            Experience the power of AI-driven helmet detection with our interactive demo
          </p>
          <button
            onClick={() => onNavigate('demo')}
            className="px-10 py-4 bg-amber-500 text-slate-900 rounded-xl font-semibold hover:bg-amber-400 transition-all shadow-xl hover:shadow-2xl text-lg hover:scale-105"
          >
            Launch Demo
          </button>
        </div>
      </section>
    </div>
  );
}