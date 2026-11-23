import React from 'react';
import { AlertTriangle, RefreshCcw } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[React ErrorBoundary]', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-screen">
          <div className="error-card glass">
            <AlertTriangle size={48} color="#ef4444" />
            <h1>Something went wrong</h1>
            <p>
              The application encountered an unexpected error. This usually happens due to
              a component crash or data inconsistency.
            </p>
            <div className="error-details">
              <code>{this.state.error?.toString()}</code>
            </div>
            <button
              className="btn-primary"
              onClick={() => window.location.reload()}
              style={{ marginTop: '24px' }}
            >
              <RefreshCcw size={18} />
              <span>Reload Application</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
