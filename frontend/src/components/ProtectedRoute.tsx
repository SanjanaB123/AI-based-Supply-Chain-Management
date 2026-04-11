import { useAuth } from '@clerk/clerk-react';
import { Navigate, Outlet } from 'react-router-dom';

export default function ProtectedRoute() {
  const { isLoaded, isSignedIn } = useAuth();

  // Wait for Clerk to finish loading the session before deciding
  if (!isLoaded) return null;

  if (!isSignedIn) return <Navigate to="/sign-in" replace />;

  return <Outlet />;
}
