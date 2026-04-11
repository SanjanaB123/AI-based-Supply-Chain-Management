/**
 * Theme-aware Clerk appearance objects for Stratos.
 *
 * Two named exports — clerkLightTheme and clerkDarkTheme — are passed to
 * ClerkProvider (via ThemedClerkProvider) based on the active app theme.
 * All Clerk surfaces (SignIn, SignUp, UserButton popup) pick up these values.
 *
 * Strategy:
 *   - card element is set to bg-transparent so the auth layout's own
 *     background shows through, giving seamless integration.
 *   - CSS variables (colorText, colorInputBackground, etc.) control
 *     Clerk's internal element colors for inputs, labels, buttons, dividers.
 *   - userButtonPopoverCard gets an explicit background so the popup
 *     floats correctly above the page in both themes.
 */

// ── Light theme ───────────────────────────────────────────────────────────────

export const clerkLightTheme = {
  variables: {
    colorPrimary: '#1d4ed8',           // blue-700 — primary actions, focus rings
    colorBackground: '#ffffff',
    colorInputBackground: '#f8fafc',   // slate-50
    colorText: '#0f172a',              // slate-900
    colorTextSecondary: '#64748b',     // slate-500
    colorInputText: '#0f172a',
    colorDanger: '#dc2626',            // red-600
    borderRadius: '0.5rem',
    fontFamily: 'inherit',
  },
  elements: {
    // Auth card — transparent so AuthLayout's white bg shows through
    card: 'shadow-none border-0 bg-transparent',

    // Header
    headerTitle:    'text-[22px] font-bold tracking-tight text-slate-900',
    headerSubtitle: 'text-sm text-slate-500',

    // Social OAuth buttons
    socialButtonsBlockButton:
      'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-300 rounded-lg font-medium text-[13px] transition-colors',
    socialButtonsBlockButtonText: 'font-medium text-[13px] text-slate-700',

    // Divider
    dividerLine: 'bg-slate-200',
    dividerText: 'text-slate-400 text-xs',

    // Form fields
    formFieldLabel: 'text-[13px] font-medium text-slate-700',
    formFieldInput:
      'border-slate-200 bg-slate-50 text-slate-900 rounded-lg text-[13px] transition-all focus:border-blue-400 focus:ring-2 focus:ring-blue-100',
    formFieldInputShowPasswordButton: 'text-slate-400 hover:text-slate-600',

    // Primary button
    formButtonPrimary:
      'bg-blue-700 text-white hover:bg-blue-800 rounded-lg font-semibold text-[13px] transition-colors shadow-none',

    // Footer
    footerActionText: 'text-[13px] text-slate-500',
    footerActionLink: 'text-blue-700 hover:text-blue-800 font-medium text-[13px]',

    // Identity preview (step-up flows)
    identityPreviewText:       'text-slate-700',
    identityPreviewEditButton: 'text-blue-700 hover:text-blue-800',

    // Alerts
    alertText: 'text-[13px] text-red-700',

    // Badge
    badge: 'text-blue-700 bg-blue-50',

    // ── UserButton popup ───────────────────────────────────────────────────
    userButtonPopoverCard:
      'bg-white border border-slate-200/80 shadow-xl rounded-xl',
    userButtonPopoverActions: 'py-1',
    userButtonPopoverActionButton:
      'hover:bg-slate-50 rounded-lg transition-colors',
    userButtonPopoverActionButtonText:
      'text-[13px] font-medium text-slate-700',
    userButtonPopoverActionButtonIcon: 'text-slate-400',
    userButtonPopoverFooter:
      'border-t border-slate-100',
    userPreviewMainIdentifier:
      'text-[14px] font-semibold text-slate-900',
    userPreviewSecondaryIdentifier: 'text-[12px] text-slate-500',
  },
};

// ── Dark theme ────────────────────────────────────────────────────────────────

export const clerkDarkTheme = {
  variables: {
    colorPrimary: '#60a5fa',           // blue-400 — more legible on dark surfaces
    colorBackground: '#0f172a',        // slate-950 — used internally by Clerk
    colorInputBackground: '#1e293b',   // slate-800
    colorText: '#f1f5f9',              // slate-100
    colorTextSecondary: '#94a3b8',     // slate-400
    colorInputText: '#f1f5f9',
    colorDanger: '#f87171',            // red-400
    borderRadius: '0.5rem',
    fontFamily: 'inherit',
  },
  elements: {
    // Auth card — transparent; AuthLayout dark:bg-slate-950 shows through
    card: 'shadow-none border-0 bg-transparent',

    // Header
    headerTitle:    'text-[22px] font-bold tracking-tight text-slate-100',
    headerSubtitle: 'text-sm text-slate-400',

    // Social OAuth buttons — use ! to override Clerk's hardcoded color styles
    socialButtonsBlockButton:
      'border! border-slate-700! bg-slate-800! text-slate-200! hover:bg-slate-700! hover:border-slate-600! rounded-lg font-medium text-[13px] transition-colors',
    socialButtonsBlockButtonText: 'font-medium text-[13px] text-slate-200!',

    // Divider
    dividerLine: 'bg-slate-700',
    dividerText: 'text-slate-500 text-xs',

    // Form fields
    formFieldLabel: 'text-[13px] font-medium text-slate-300',
    formFieldInput:
      'border-slate-700 bg-slate-800 text-slate-100 rounded-lg text-[13px] transition-all focus:border-blue-500 focus:ring-2 focus:ring-blue-900/40',
    formFieldInputShowPasswordButton: 'text-slate-500 hover:text-slate-300',

    // Primary button
    formButtonPrimary:
      'bg-blue-600 text-white hover:bg-blue-500 rounded-lg font-semibold text-[13px] transition-colors shadow-none',

    // Footer
    footerActionText: 'text-[13px] text-slate-400',
    footerActionLink: 'text-blue-400 hover:text-blue-300 font-medium text-[13px]',

    // Identity preview
    identityPreviewText:       'text-slate-300',
    identityPreviewEditButton: 'text-blue-400 hover:text-blue-300',

    // Alerts
    alertText: 'text-[13px] text-red-400',

    // Badge
    badge: 'text-blue-400 bg-blue-900/30',

    // ── UserButton popup ───────────────────────────────────────────────────
    userButtonPopoverCard:
      'bg-slate-800 border border-slate-700/80 shadow-xl rounded-xl',
    userButtonPopoverActions: 'py-1',
    userButtonPopoverActionButton:
      'hover:bg-slate-700 rounded-lg transition-colors',
    userButtonPopoverActionButtonText:
      'text-[13px] font-medium text-slate-200!',
    userButtonPopoverActionButtonIcon: 'text-slate-400',
    userButtonPopoverFooter:
      'border-t border-slate-700',
    userPreviewMainIdentifier:
      'text-[14px] font-semibold text-slate-100',
    userPreviewSecondaryIdentifier: 'text-[12px] text-slate-400',
  },
};
