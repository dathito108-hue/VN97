package ai.vn97.app

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** Presentation only: capability gates and action listeners remain in the activity. */
internal class VN97Dashboard(private val context: Context) {
    private val canvasColor = Color.rgb(10, 17, 29)
    private val surface = Color.rgb(20, 32, 49)
    private val ink = Color.rgb(235, 242, 252)
    private val muted = Color.rgb(167, 187, 209)
    private val accent = Color.rgb(91, 224, 205)
    private fun dp(value: Int) = (value * context.resources.displayMetrics.density).toInt()
    private fun column() = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
    private fun shape(color: Int) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(16).toFloat()
    }
    private fun detach(view: View) { (view.parent as? ViewGroup)?.removeView(view) }
    private fun add(parent: LinearLayout, view: View) {
        detach(view)
        parent.addView(view, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8) })
    }
    private fun text(value: String, size: Float, color: Int) = TextView(context).apply {
        text = value
        textSize = size
        setTextColor(color)
    }
    private fun style(view: View) {
        when (view) {
            is Button -> {
                view.isAllCaps = false
                view.textSize = 14f
                view.minHeight = dp(48)
                view.setPadding(dp(12), dp(8), dp(12), dp(8))
                view.background = shape(Color.rgb(32, 53, 73))
                view.setTextColor(ColorStateList(
                    arrayOf(intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
                    intArrayOf(Color.rgb(115, 132, 152), ink),
                ))
            }
            is EditText -> {
                view.setTextColor(ink)
                view.setHintTextColor(muted)
                view.backgroundTintList = ColorStateList.valueOf(accent)
            }
            is TextView -> {
                view.setTextColor(ink)
                view.setLineSpacing(dp(3).toFloat(), 1f)
                view.textSize = 14f
            }
            is LinearLayout -> {
                // Narrow phones and large fonts need full-width action rows.
                view.orientation = LinearLayout.VERTICAL
                for (i in 0 until view.childCount) {
                    val child = view.getChildAt(i)
                    style(child)
                    child.layoutParams = LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8) }
                }
            }
        }
    }

    fun card(title: String, description: String, vararg controls: View): View = column().apply {
        background = shape(surface)
        setPadding(dp(16), dp(16), dp(16), dp(8))
        add(this, text(title, 18f, ink).apply { setTypeface(null, Typeface.BOLD) })
        add(this, text(description, 13f, muted))
        controls.forEach { style(it); add(this, it) }
    }

    fun build(
        avatar: View,
        status: TextView,
        composer: LinearLayout,
        approvals: List<View>,
        pages: List<List<View>>,
    ): View {
        val shell = column().apply {
            setBackgroundColor(canvasColor)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        val header = LinearLayout(context).apply { gravity = Gravity.CENTER_VERTICAL }
        val heading = column().apply {
            add(this, text("VN97", 24f, ink).apply { setTypeface(null, Typeface.BOLD) })
            add(this, text("Trợ lý trên điện thoại", 12f, muted))
        }
        header.addView(heading, LinearLayout.LayoutParams(0, -2, 1f))
        detach(avatar)
        // Fixed header: the translucent GL surface never scrolls across text.
        header.addView(avatar, LinearLayout.LayoutParams(dp(64), dp(64)))
        shell.addView(header)

        val tabs = LinearLayout(context)
        shell.addView(tabs, LinearLayout.LayoutParams(-1, -2))
        val content = column()
        val scroll = ScrollView(context).apply {
            isFillViewport = true
            addView(content)
        }
        shell.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        style(status)
        status.setPadding(dp(8), dp(10), dp(8), dp(10))
        status.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        add(content, status)
        // Approvals stay outside page visibility, including during background updates.
        approvals.forEach { style(it); add(content, it) }
        val panels = pages.map { cards -> column().apply { cards.forEach { add(this, it) } } }
        panels.forEachIndexed { index, panel ->
            add(content, panel)
            panel.visibility = if (index == 0) View.VISIBLE else View.GONE
        }
        val navigation = listOf("Trò chuyện", "Công cụ", "Hệ thống").map { label ->
            Button(context).apply { text = label; style(this); textSize = 12f }
        }
        fun select(index: Int) {
            panels.forEachIndexed { i, panel -> panel.visibility = if (i == index) View.VISIBLE else View.GONE }
            navigation.forEachIndexed { i, button ->
                button.isSelected = i == index
                button.background = shape(if (i == index) Color.rgb(25, 83, 84) else surface)
                button.setTextColor(if (i == index) accent else muted)
            }
            scroll.scrollTo(0, 0)
        }
        navigation.forEachIndexed { index, button ->
            tabs.addView(button, LinearLayout.LayoutParams(0, -2, 1f).apply {
                setMargins(dp(2), dp(4), dp(2), dp(8))
            })
            button.setOnClickListener { select(index) }
        }
        select(0)
        detach(composer)
        // Keep the composer reachable even with a long transcript.
        for (i in 0 until composer.childCount) style(composer.getChildAt(i))
        composer.gravity = Gravity.CENTER_VERTICAL
        shell.addView(composer, LinearLayout.LayoutParams(-1, -2))
        return shell
    }
}
