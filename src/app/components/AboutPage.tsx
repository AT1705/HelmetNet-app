import { Brain, Shield, Users } from 'lucide-react';
import { ImageWithFallback } from '@/app/components/figma/ImageWithFallback';

export function AboutPage() {
  const experiments = [
    {
      number: 1,
      title: 'Poor cap detection (baseline limitations)',
      issue: 'The model frequently confused caps / head coverings with helmets, producing poor discrimination.',
      learning: 'Model performance is bottlenecked by labeling policy quality more than raw architecture.'
    },
    {
      number: 2,
      title: 'Helmet type classification refinement',
      issue: 'System struggled to distinguish between different helmet types and safety standards.',
      learning: 'Dataset diversity across helmet types and viewing angles is critical for robust detection.'
    },
    {
      number: 3,
      title: 'Real-time stream optimization',
      issue: 'Processing latency exceeded acceptable thresholds for live traffic monitoring.',
      learning: 'Architecture optimization and edge computing integration essential for real-time applications.'
    },
    {
      number: 4,
      title: 'Multi-angle detection enhancement',
      issue: 'Detection accuracy dropped significantly for side and rear viewing angles.',
      learning: 'Comprehensive multi-angle dataset coverage ensures consistent performance across deployment scenarios.'
    }
  ];

  const complianceReferences = [
    {
      title: 'Section 119(2) Road Transport Act 1987',
      description: 'Non-compliance signaling and legal framework for helmet enforcement'
    },
    {
      title: 'SIRIM MS 1:2011',
      description: 'Helmet compliance standard reference for safety certification'
    }
  ];

  return (
    <div className="min-h-screen bg-slate-50 pt-16">
      {/* Hero Section */}
      <section className="py-20 bg-gradient-to-br from-slate-800 to-slate-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl">
            <h1 className="text-5xl font-bold mb-6">About HelmetNet</h1>
            <p className="text-lg text-gray-300">
              A Computer Vision pipeline designed to detect helmet compliance for motorcycle riders through iterative research and development.
            </p>
          </div>
        </div>
      </section>

      {/* About HelmetNet - Single Unified Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white rounded-xl shadow-xl overflow-hidden border border-slate-200">
            <div className="p-8 md:p-12">
              <h2 className="text-4xl font-bold mb-8 text-slate-900">About HelmetNet</h2>
              
              <div className="grid lg:grid-cols-2 gap-8 mb-12">
                <div className="space-y-6">
                  <p className="text-lg text-gray-700 leading-relaxed">
                    HelmetNet is a Computer Vision pipeline designed to detect helmet compliance for motorcycle riders. This portal demonstrates how model quality evolves through iterative dataset labeling, class definitions, and annotation discipline.
                  </p>
                  <p className="text-lg text-gray-700 leading-relaxed">
                    The system supports inference across <span className="font-semibold text-slate-800">Images</span>, <span className="font-semibold text-slate-800">Videos</span>, and <span className="font-semibold text-slate-800">Real-time streams</span>.
                  </p>
                  <div className="bg-slate-50 p-6 rounded-xl border-l-4 border-amber-500">
                    <p className="text-gray-700 leading-relaxed">
                      The redesign you are viewing emphasizes a professional "government portal" experience: guided navigation, clean configuration, and compliance-oriented insights.
                    </p>
                  </div>
                </div>
                <div className="relative h-[350px] rounded-xl overflow-hidden shadow-lg border border-slate-200">
                  <ImageWithFallback
                    src="https://images.unsplash.com/photo-1569932353341-b518d82f8a54?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb3RvcmN5Y2xlJTIwc2FmZXR5JTIwdGVjaG5vbG9neXxlbnwxfHx8fDE3Njg0ODI1NTd8MA&ixlib=rb-4.1.0&q=80&w=1080"
                    alt="Motorcycle safety technology"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>

              {/* Compliance Orientation */}
              <div className="mb-12">
                <h3 className="text-2xl font-semibold mb-6 text-slate-900">Compliance Orientation</h3>
                <p className="text-gray-700 mb-6 leading-relaxed">
                  HelmetNet is positioned as a compliance-support tool rather than a purely technical demo. The portal integrates prescriptive guidance referencing:
                </p>
                <div className="grid md:grid-cols-2 gap-6">
                  {complianceReferences.map((ref, index) => (
                    <div key={index} className="bg-slate-50 p-6 rounded-xl border-l-4 border-slate-700">
                      <h4 className="font-semibold text-lg mb-2 text-slate-800">{ref.title}</h4>
                      <p className="text-gray-600">{ref.description}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Technology Overview */}
              <div>
                <h3 className="text-2xl font-semibold mb-6 text-slate-900">Technology Stack</h3>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 hover:shadow-lg transition-shadow">
                    <Brain className="size-10 text-slate-700 mb-3" />
                    <h4 className="font-semibold mb-2 text-slate-900">Deep Learning</h4>
                    <p className="text-sm text-gray-600">
                      CNN-based models trained for helmet detection with 99.2% accuracy
                    </p>
                  </div>
                  <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 hover:shadow-lg transition-shadow">
                    <Shield className="size-10 text-slate-700 mb-3" />
                    <h4 className="font-semibold mb-2 text-slate-900">Computer Vision</h4>
                    <p className="text-sm text-gray-600">
                      Multi-angle detection with real-time image processing capabilities
                    </p>
                  </div>
                  <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 hover:shadow-lg transition-shadow">
                    <Users className="size-10 text-slate-700 mb-3" />
                    <h4 className="font-semibold mb-2 text-slate-900">Edge Computing</h4>
                    <p className="text-sm text-gray-600">
                      Sub-second detection time with local processing for privacy
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* The 4 Experiments - Minimalist Grid */}
      <section className="py-16 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold mb-4 text-slate-900">The 4 Experiments</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Research progression addressing concrete failure modes and refining the dataset
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            {experiments.map((exp, index) => (
              <div key={index} className="bg-white rounded-xl p-6 border border-slate-200 hover:shadow-xl transition-all duration-300">
                <div className="flex items-start gap-4 mb-4">
                  <div className="flex-shrink-0 size-12 bg-gradient-to-br from-slate-700 to-slate-800 text-white rounded-xl flex items-center justify-center font-bold text-xl shadow-md">
                    E{exp.number}
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900 pt-2">{exp.title}</h3>
                </div>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="font-semibold text-slate-800">Issue:</span>
                    <p className="text-gray-600 mt-1">{exp.issue}</p>
                  </div>
                  <div>
                    <span className="font-semibold text-slate-800">Learning:</span>
                    <p className="text-gray-700 mt-1">{exp.learning}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}