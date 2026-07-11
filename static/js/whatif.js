// What-if explorer: adjust inputs, re-run prediction via /whatif, show the delta.

(function () {
  const base = window.BASE_INPUT;
  const baseRent = window.BASE_RENT;

  const bhk = document.getElementById("wf_bhk");
  const furnishing = document.getElementById("wf_furnishing");
  const area = document.getElementById("wf_area");
  const metro = document.getElementById("wf_metro");

  const bhkLabel = document.getElementById("bhkLabel");
  const areaLabel = document.getElementById("areaLabel");
  const rentOut = document.getElementById("wf_rent");
  const deltaOut = document.getElementById("wf_delta");

  // initialise controls from the base input
  bhk.value = base.bhk || 2;
  area.value = base.carpet_area || 800;
  furnishing.value = base.furnishing || "Semi-Furnished";
  metro.value = base.has_metro ? "1" : (base.metro_mins && base.metro_mins <= 30 ? "1" : "0");
  bhkLabel.textContent = bhk.value;
  areaLabel.textContent = area.value;

  let timer = null;

  function fmt(n) {
    return "₹" + Math.round(n).toLocaleString("en-IN");
  }

  async function recompute() {
    bhkLabel.textContent = bhk.value;
    areaLabel.textContent = area.value;

    const changes = {
      bhk: parseInt(bhk.value, 10),
      carpet_area: parseFloat(area.value),
      furnishing: furnishing.value,
      // metro: yes -> a small metro time; no -> null (pipeline sets 'far')
      metro_mins: metro.value === "1" ? 5 : null,
    };

    try {
      const res = await fetch("/whatif", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base_input: base, changes: changes }),
      });
      const data = await res.json();
      const newRent = data.rent;

      rentOut.textContent = fmt(newRent);

      const diff = newRent - baseRent;
      if (Math.abs(diff) < 1) {
        deltaOut.textContent = "Same as your original estimate";
        deltaOut.className = "delta";
      } else if (diff > 0) {
        deltaOut.textContent = "+" + fmt(diff) + " more than your estimate";
        deltaOut.className = "delta up";
      } else {
        deltaOut.textContent = fmt(diff) + " (you'd save " + fmt(-diff) + ")";
        deltaOut.className = "delta down";
      }
    } catch (e) {
      deltaOut.textContent = "Could not compute — try again";
      deltaOut.className = "delta";
    }
  }

  // debounce so dragging a slider doesn't fire dozens of requests
  function onChange() {
    bhkLabel.textContent = bhk.value;
    areaLabel.textContent = area.value;
    clearTimeout(timer);
    timer = setTimeout(recompute, 250);
  }

  [bhk, area].forEach((el) => el.addEventListener("input", onChange));
  [furnishing, metro].forEach((el) => el.addEventListener("change", recompute));
})();
