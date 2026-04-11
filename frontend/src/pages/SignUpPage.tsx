import { SignUp } from '@clerk/clerk-react';

export default function SignUpPage() {
  return (
    <SignUp
      routing="path"
      path="/sign-up"
      signInUrl="/sign-in"
      fallbackRedirectUrl="/dashboard"
    />
  );
}
