# FRONTEND_RULES.md

## Purpose

This file defines the UI, UX, architecture, styling, and code quality rules for the `frontend/` app.

Every implementation prompt should follow these rules unless a later prompt explicitly overrides a specific item.

---

## Product Direction

- Brand name: **Stratos**
- Product type: premium AI-powered supply chain intelligence platform
- Feel: modern, credible, product-grade SaaS
- Avoid: student-project UI, generic admin template look, random colors, cramped layouts, inconsistent spacing
- Primary target: desktop and laptop screens
- Secondary target: tablet widths down to **iPad mini**
- Mobile-first optimization is **not required** right now

---

## Core Visual Principles

1. Build for clarity first, polish second.
2. Use color intentionally to communicate meaning, not decoration.
3. Keep layouts spacious and breathable.
4. Prefer a small number of strong sections over many weak sections.
5. Keep interaction patterns consistent across all pages.
6. Preserve a single visual language across:
   - landing page
   - auth pages
   - dashboard
   - future chat widget
7. Do not introduce a second design system later. Build on the same one throughout.

---

## Color System

### Base neutrals
Use a restrained neutral foundation:

- **Background**: very light gray or warm-neutral off-white
- **Surface cards**: white or near-white
- **Primary text**: deep navy/slate
- **Secondary text**: muted slate/gray
- **Borders**: soft neutral gray
- **Hover states**: slightly darker or lighter tone of the existing surface, never a random new color

### Brand color
Use a deep, premium blue/navy for the primary brand feel.

Recommended direction:
- primary brand tone: navy or deep indigo
- primary action buttons: dark navy background, white text
- focus ring and selected states: blue with accessible contrast

### Semantic colors
These must stay consistent everywhere:

- **Critical**: red
- **Low / Warning**: amber / orange
- **Healthy / Good**: green
- **Info / Interactive emphasis**: blue

Do not swap these meanings across pages.

### Semantic usage rules
- KPI cards may use soft tinted backgrounds with stronger colored text
- Charts must use the same semantic colors as the rest of the app
- Table chips/badges must match chart colors
- Do not use bright rainbow palettes
- Do not use saturated gradients unless explicitly requested later

---

## Typography Rules

### Overall approach
- Clean, strong hierarchy
- Large, confident page titles
- Medium-weight section headers
- Smaller muted supporting text

### Hierarchy
Use roughly this structure consistently:

- **Page title**: largest on page, bold
- **Section title**: clearly distinct, medium-large
- **Card metric value**: large and bold
- **Body text**: readable and not cramped
- **Secondary/supporting text**: smaller, muted
- **Micro labels**: small but still legible

### Typography rules
- Avoid overly tiny text
- Avoid too many font weights
- Prefer 3–4 text sizes used consistently
- Do not make dashboard copy verbose
- Keep labels concise
- Use sentence case unless a strong reason exists otherwise

---

## Layout Rules

### Global layout
- Use a centered content container on major pages
- Prefer generous horizontal padding
- Use consistent top/bottom spacing between major sections
- Avoid edge-to-edge crowded layouts on large screens

### Dashboard layout
- Desktop-first
- Important information should appear above the fold
- KPI summary should be visible quickly
- Content should scale well on wide screens
- Must remain usable at iPad mini width without collapsing into chaos

### Auth pages
- Keep Clerk working first
- Surround Clerk forms with branded structure
- Do not leave them as isolated floating widgets forever
- Final auth styling should still feel part of Stratos

### Landing page
- Full-height or near full-height hero structure is preferred
- Clean and minimal
- Strong headline and CTA clarity
- Avoid long-scroll marketing sections unless explicitly added later

---

## Spacing Rules

- Prefer spacious spacing over dense spacing
- Use a consistent spacing scale
- Section padding should be visually generous
- Cards should breathe internally
- Do not stack unrelated elements too tightly
- Keep horizontal gaps aligned between cards and sections

### Practical spacing guidance
- Use consistent gap classes
- Reuse the same spacing rhythm in:
  - cards
  - nav bars
  - auth wrappers
  - charts
  - tables

---

## Component Rules

### General
- Components must be reusable when reuse is obvious
- Do not create abstract wrappers for one-time use unless they clearly improve structure
- Prefer small, readable components
- Each component should have one job

### Required style consistency
Shared patterns should become shared components when repeated:
- buttons
- cards
- section shells
- badges/chips
- loading skeletons
- tab controls
- store selector

### Component design
- Rounded corners should be consistent
- Card borders/shadows should be subtle
- Hover states should be gentle
- Focus states must be visible and accessible

---

## Dashboard UX Rules

### Information architecture
- Summary first
- Detail second
- Advanced exploration third

### KPI cards
- Use semantic color carefully
- Metric values must be easy to scan
- Supporting labels should explain what is being measured
- Do not overload each card with too much text

### Charts
- Charts must feel integrated, not like demo snippets
- Always provide labels, legend, and context
- Do not leave a chart unexplained
- Use semantic colors consistently
- Keep chart containers visually aligned with card design
- Prefer simple, readable charts over flashy ones

### Tables
- Tables should be clean and product-grade
- Align numeric columns properly
- Avoid excessive borders
- Use badges/chips for status
- Support scanning with spacing and emphasis, not noise

### Tabs / section switching
- Selected state must be obvious
- Inactive state should still look polished
- Keep the tab style consistent across modules

---

## Loading, Empty, and Error States

### Loading
- Never leave blank white screens while data is loading
- Use skeletons where layout is known
- Skeletons should match the final structure
- Use spinners only where skeletons are not appropriate

### Empty states
- Empty states should be calm and informative
- Do not imply failure when there is simply no data

### Error states
- Errors must be visible but not visually aggressive
- Explain what failed in plain language
- Offer retry if appropriate

---

## Chat Widget Rules (for later)

- Bottom-right anchored
- Expandable like modern chat panels
- Must look like part of the product, not a third-party popup
- Minimized and expanded states should both feel polished
- Preserve semantic consistency with the dashboard
- If chat is unavailable, show a graceful disabled state using `/api/ai-status`

---

## Responsiveness Rules

### Target behavior
- Optimize for:
  - large desktop
  - laptop
  - tablet / iPad mini

### Rules
- Do not design for phone-first layouts right now
- Avoid complex multi-column layouts that break too early
- Collapse sections thoughtfully
- Preserve hierarchy as width shrinks
- Critical dashboard actions should remain visible on tablet widths

---

## Animation Rules

- No animation by default unless it serves UX
- Prefer subtle transitions
- Avoid over-animated dashboards
- GSAP may be added later for marketing polish only
- Do not add decorative animation to data UI unless explicitly asked

---

## Accessibility Rules

- Maintain strong text contrast
- Interactive elements must have visible focus states
- Color must not be the only indicator of meaning
- Add labels/tooltips where meaning could be ambiguous
- Keep clickable areas sufficiently large
- Do not rely on placeholder text alone for form understanding

---

## Code Architecture Rules

### General
- Use TypeScript strictly and correctly
- Keep files small and focused
- Prefer explicit, readable code over clever abstractions
- Avoid deep nesting in components

### File organization
Place code in the existing structure unless a prompt requires expansion:

- `src/app/` -> layouts and app-level wrappers
- `src/components/` -> reusable UI components
- `src/pages/` -> route-level pages
- `src/lib/` -> API helpers, config, utilities
- `src/routes/` -> router setup
- `src/types/` -> shared API and domain types
- `src/hooks/` -> reusable hooks when clearly justified

### Reusability rules
- Extract repeated UI only after the pattern is clearly repeated
- Do not create utils for trivial one-line logic
- Keep API logic separate from presentational UI when possible

---

## TypeScript Rules

- No `any` unless truly unavoidable and documented
- Prefer explicit interfaces/types for backend responses
- Narrow unions where possible
- Handle `undefined` and loading states safely
- Do not suppress errors unless there is a strong reason
- Avoid unsafe type assertions unless they are necessary and explained

### API typing
- Every backend route used by the frontend should have a corresponding type
- Keep response types close to the API layer or in `src/types/`
- Do not duplicate identical types in multiple files

---

## Error Prevention Rules

- App must run without TypeScript errors
- Avoid unused imports and dead code
- Do not leave commented-out blocks unless explicitly useful
- Remove obsolete template boilerplate
- Keep README and folder tree updated after each major prompt

---

## Styling Rules

### Tailwind
- Use Tailwind consistently
- Prefer readable class groupings
- Avoid giant unreadable class strings if a reusable component can solve it
- Keep spacing, color, and typography choices consistent

### Repetition
If the same class combination appears repeatedly, consider extracting a component.

### Avoid
- random one-off colors
- multiple different border radius systems
- inconsistent padding patterns
- mismatched button styles

---

## State Management Rules

- Use local React state unless broader state is clearly needed
- Do not introduce heavy state management libraries unless justified
- Keep data-fetching state near the relevant page/module at first
- Abstract only when the usage pattern becomes clear

---

## API Integration Rules

- Always use the shared authenticated API helper for protected routes
- Always attach Clerk Bearer token for protected backend calls
- Use Clerk `userId` as the future `/api/chat` username/thread identifier
- Surface loading and error states clearly in the UI
- Do not hardcode backend data when the live backend works

---

## README / Documentation Rules

After each major implementation prompt:
- update `frontend/README.md`
- summarize what changed
- include new files created/modified
- include updated folder tree if structure changed
- include verification steps
- include any manual env/setup instructions

---

## What to Avoid

Do not produce:
- grayscale-only bland UI
- random startup neon palette
- generic dashboard boilerplate
- cramped layouts
- inconsistent auth/dashboard/landing page styling
- overcomplicated abstractions
- loosely typed API code
- blank loading screens

---

## High-Level Quality Standard

Every new frontend step should aim for:

- clear structure
- strict typing
- visual consistency
- polished but restrained UI
- meaningful color
- scalable component design
- stable integration with the deployed backend
- readiness for future expansion without rewrite
