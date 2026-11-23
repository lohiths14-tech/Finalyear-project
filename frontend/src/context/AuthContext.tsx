import React, { createContext, useContext, useState, useEffect } from 'react';
import apiClient from '../services/api';

interface AuthContextType {
  apiKey: string | null;
  login: (key: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [apiKey, setApiKey] = useState<string | null>(null);

  useEffect(() => {
    const storedKey = localStorage.getItem('agripulse_api_key');
    if (storedKey) {
      setApiKey(storedKey);
      apiClient.defaults.headers.common['X-API-Key'] = storedKey;
    }
  }, []);

  const login = (key: string) => {
    setApiKey(key);
    // Don't persist or send the no-auth placeholder as a real header
    if (key === '__no_auth__') {
      localStorage.removeItem('agripulse_api_key');
      delete apiClient.defaults.headers.common['X-API-Key'];
    } else {
      localStorage.setItem('agripulse_api_key', key);
      apiClient.defaults.headers.common['X-API-Key'] = key;
    }
  };

  const logout = () => {
    setApiKey(null);
    localStorage.removeItem('agripulse_api_key');
    delete apiClient.defaults.headers.common['X-API-Key'];
  };

  return (
    <AuthContext.Provider value={{ apiKey, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
