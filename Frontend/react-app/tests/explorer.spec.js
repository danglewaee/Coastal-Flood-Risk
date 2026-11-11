import { test, expect } from "@playwright/test";

test("scenario navigation, map selection and export use the real API", async ({
  page,
}, testInfo) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?city=boston&scenario=plus_50cm");
  await expect(
    page.getByRole("heading", { name: "Boston Harbor, MA", exact: true }),
  ).toBeVisible({ timeout: 45000 });
  await expect(page.locator(".leaflet-interactive").first()).toBeVisible();
  await expect(page.locator(".analysis-pane")).not.toContainText(
    "Operational Summary",
  );
  await page.getByRole("link", { name: "Hotspots", exact: true }).click();
  await expect(page).toHaveURL(/\/hotspots/);
  await page
    .getByRole("button", { name: /Inspect region/ })
    .first()
    .click();
  await expect(
    page.getByRole("complementary", { name: "Selected flood region" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Export selected region" }),
  ).toBeVisible({ timeout: 30000 });
  await expect(page).toHaveURL(/region=/);
  await page.getByRole("button", { name: "Close region inspector" }).click();
  await page.getByRole("button", { name: "+100 cm", exact: true }).click();
  await expect(page).toHaveURL(/scenario=plus_100cm/);
  await expect(page.locator(".map-loading")).toHaveCount(0, { timeout: 45000 });
  await page.getByRole("link", { name: "Exposure", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Exposure", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("textbox", { name: "Search exposure layers" })
    .fill("Road");
  await expect(page.locator(".exposure-row")).toHaveCount(1);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export flood GeoJSON" }).click();
  expect((await download).suggestedFilename()).toBe(
    "boston_plus_100cm.geojson",
  );
  await page.getByRole("link", { name: "City spotlight", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Boston Harbor, MA", exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: `../../artifacts/explorer-${testInfo.project.name}.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

test("direct page links retain scenario and controls are keyboard accessible", async ({
  page,
}) => {
  await page.goto("/operations?city=boston&scenario=plus_50cm");
  await expect(
    page.getByRole("heading", { name: "Operations", exact: true }),
  ).toBeVisible({ timeout: 45000 });
  await page.getByRole("button", { name: "Map layers", exact: true }).click();
  const infrastructure = page.getByRole("checkbox", {
    name: "Affected infrastructure",
  });
  await infrastructure.check();
  await expect(infrastructure).toBeChecked();
  await page
    .getByRole("checkbox", { name: "Flood extent", exact: true })
    .uncheck();
  await expect(
    page.getByRole("checkbox", { name: "Flood extent", exact: true }),
  ).not.toBeChecked();
  await page.getByRole("button", { name: "Map layers", exact: true }).click();
  await page.getByRole("link", { name: "Forecast", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Water levels" }),
  ).toBeVisible();
  await expect(page.getByLabel("Evaluation run")).toBeVisible();
});
