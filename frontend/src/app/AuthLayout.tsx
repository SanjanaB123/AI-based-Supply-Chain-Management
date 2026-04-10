import { Outlet } from 'react-router-dom';

export default function AuthLayout() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-50 px-4">
      <div className="mb-6 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-gray-900">Stratos</h1>
        <p className="mt-1 text-sm text-gray-500">AI-powered supply chain intelligence</p>
      </div>
      <Outlet />
    </div>
  );
}
