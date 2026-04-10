/**
 * Shared Clerk appearance config for Stratos auth pages.
 *
 * Applied to both <SignIn> and <SignUp> to align Clerk's default UI
 * with the Stratos design language: slate neutrals, blue accents,
 * clean typography, consistent border radius.
 */
const clerkAppearance = {
  variables: {
    colorPrimary: '#1e40af',           // blue-800 — primary actions + focus rings
    colorBackground: '#ffffff',
    colorInputBackground: '#f8fafc',   // slate-50
    colorText: '#0f172a',              // slate-900
    colorTextSecondary: '#64748b',     // slate-500
    colorInputText: '#0f172a',
    colorDanger: '#dc2626',            // red-600
    borderRadius: '0.5rem',            // rounded-lg — matches app card radius
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif',
  },
  elements: {
    // Remove default card shadow/border so the form sits flush on the panel
    card: 'shadow-none border-0 bg-transparent',

    // Typography
    headerTitle: 'text-[22px] font-bold tracking-tight text-slate-900',
    headerSubtitle: 'text-sm text-slate-500',

    // Social OAuth buttons
    socialButtonsBlockButton:
      'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-300 rounded-lg font-medium text-[13px] transition-colors',
    socialButtonsBlockButtonText: 'font-medium text-[13px] text-slate-700',

    // Divider ("or")
    dividerLine: 'bg-slate-200',
    dividerText: 'text-slate-400 text-xs',

    // Form fields
    formFieldLabel: 'text-[13px] font-medium text-slate-700',
    formFieldInput:
      'border-slate-200 bg-slate-50 text-slate-900 rounded-lg text-[13px] transition-all focus:border-blue-400 focus:ring-2 focus:ring-blue-100',
    formFieldInputShowPasswordButton: 'text-slate-400 hover:text-slate-600',

    // Primary submit button
    formButtonPrimary:
      'bg-blue-800 text-white hover:bg-blue-900 rounded-lg font-semibold text-[13px] transition-colors',

    // Footer links
    footerActionText: 'text-[13px] text-slate-500',
    footerActionLink: 'text-blue-700 hover:text-blue-800 font-medium text-[13px]',

    // Identity preview (step-up flows)
    identityPreviewText: 'text-slate-700',
    identityPreviewEditButton: 'text-blue-700 hover:text-blue-800',

    // Alerts / error messages
    alertText: 'text-[13px] text-red-700',

    // Badge (e.g. "Verified")
    badge: 'text-blue-700 bg-blue-50',
  },
};

export default clerkAppearance;
